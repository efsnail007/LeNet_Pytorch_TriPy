import time

import tensorrt as trt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)


test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=test_transform
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True
)


def build_engine_from_onnx(onnx_path: str, engine_path: str, fp16: bool = False):
    logger = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError("ONNX parse failed:\n" + msgs)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "build_serialized_network вернул None (не смог собрать engine)."
        )

    with open(engine_path, "wb") as f:
        f.write(serialized)


build_engine_from_onnx("lenet_cifar10.onnx", "lenet_cifar10_fp32.trt", fp16=False)
build_engine_from_onnx("lenet_cifar10.onnx", "lenet_cifar10_fp16.trt", fp16=True)


criterion = nn.CrossEntropyLoss()
logger = trt.Logger(trt.Logger.WARNING)


def load_trt_context(engine_path: str):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


def trt_dtype_to_torch(dt: trt.DataType):
    if dt == trt.DataType.FLOAT:
        return torch.float32
    if dt == trt.DataType.HALF:
        return torch.float16
    if dt == trt.DataType.INT32:
        return torch.int32
    if dt == trt.DataType.BOOL:
        return torch.bool
    raise NotImplementedError(dt)


def get_io_names(engine: trt.ICudaEngine):
    inputs, outputs = [], []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(name)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"Ожидал 1 вход и 1 выход, получил {len(inputs)} / {len(outputs)}"
        )
    return inputs[0], outputs[0]


def trt_forward_only(
    engine, context, input_name, output_name, images_cuda: torch.Tensor
):

    want_dtype = trt_dtype_to_torch(engine.get_tensor_dtype(input_name))
    if images_cuda.dtype != want_dtype:
        images_cuda = images_cuda.to(want_dtype)

    context.set_input_shape(input_name, tuple(images_cuda.shape))
    out_shape = tuple(context.get_tensor_shape(output_name))
    if any(d == -1 for d in out_shape):
        out_shape = (images_cuda.shape[0], out_shape[-1])
    out_dtype = trt_dtype_to_torch(engine.get_tensor_dtype(output_name))
    logits = torch.empty(out_shape, device="cuda", dtype=out_dtype)

    context.set_tensor_address(input_name, images_cuda.data_ptr())
    context.set_tensor_address(output_name, logits.data_ptr())

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    if not ok:
        raise RuntimeError("execute_async_v3 failed")
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return logits, (t1 - t0)


@torch.no_grad()
def evaluate_trt_like_pytorch(
    engine_path: str, test_loader, warmup_batches=5, tag="TensorRT"
):
    engine, context = load_trt_context(engine_path)
    input_name, output_name = get_io_names(engine)

    total = correct = 0
    run_loss = 0.0

    fwd_sum = 0.0
    e2e_sum = 0.0
    measured = 0
    bs_seen = None

    for bi, (images_cpu, labels_cpu) in enumerate(test_loader):

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        images = images_cpu.to("cuda", non_blocking=True)
        logits, _ = trt_forward_only(engine, context, input_name, output_name, images)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        e2e_dt = t1 - t0

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _ = trt_forward_only(engine, context, input_name, output_name, images)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        fwd_dt = t3 - t2

        labels = labels_cpu.to("cuda", non_blocking=True)
        loss = criterion(logits.float(), labels)
        run_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if bs_seen is None:
            bs_seen = labels.size(0)

        if bi >= warmup_batches:
            fwd_sum += fwd_dt
            e2e_sum += e2e_dt
            measured += 1

    loss = run_loss / total
    acc = correct / total

    if measured > 0 and bs_seen:
        fwd_t_batch = fwd_sum / measured
        fwd_t_img = fwd_t_batch / bs_seen
        fwd_fps = 1.0 / fwd_t_img

        e2e_t_batch = e2e_sum / measured
        e2e_t_img = e2e_t_batch / bs_seen
        e2e_fps = 1.0 / e2e_t_img
    else:
        fwd_t_batch = fwd_t_img = fwd_fps = float("nan")
        e2e_t_batch = e2e_t_img = e2e_fps = float("nan")

    print(
        f"{tag}: loss={loss:.4f} acc={acc:.4f} | "
        f"forward: {fwd_t_batch*1000:.3f} ms/batch, {fwd_t_img*1000:.4f} ms/img, {fwd_fps:.1f} FPS | "
        f"e2e: {e2e_t_batch*1000:.3f} ms/batch, {e2e_t_img*1000:.4f} ms/img, {e2e_fps:.1f} FPS"
    )

    return {
        "loss": loss,
        "acc": acc,
        "fwd_batch_s": fwd_t_batch,
        "fwd_img_s": fwd_t_img,
        "fwd_fps": fwd_fps,
        "e2e_batch_s": e2e_t_batch,
        "e2e_img_s": e2e_t_img,
        "e2e_fps": e2e_fps,
    }


KOL = 100

fwd_t_batch_sum = 0
fwd_t_img_sum = 0
fwd_fps_sum = 0
e2e_t_batch_sum = 0
e2e_t_img_sum = 0
e2e_fps_sum = 0
for _ in range(KOL):
    time.sleep(1)
    stats = evaluate_trt_like_pytorch(
        "lenet_cifar10_fp32.trt", test_loader, tag="TensorRT FP32"
    )
    fwd_t_batch_sum += stats["fwd_batch_s"]
    fwd_t_img_sum += stats["fwd_img_s"]
    fwd_fps_sum += stats["fwd_fps"]
    e2e_t_batch_sum += stats["e2e_batch_s"]
    e2e_t_img_sum += stats["e2e_img_s"]
    e2e_fps_sum += stats["e2e_fps"]

stats = {
    "fwd_batch_s": fwd_t_batch_sum / KOL,
    "fwd_img_s": fwd_t_img_sum / KOL,
    "fwd_fps": fwd_fps_sum / KOL,
    "e2e_batch_s": e2e_t_batch_sum / KOL,
    "e2e_img_s": e2e_t_img_sum / KOL,
    "e2e_fps": e2e_fps_sum / KOL,
}
print("Mean______________________________________________________________")
print(
    f"TensorRT FP32"
    f"forward: {stats['fwd_batch_s']*1000:.3f} ms/batch, {stats['fwd_img_s']*1000:.4f} ms/img, {stats['fwd_fps']:.1f} FPS | "
    f"e2e: {stats['e2e_batch_s']*1000:.3f} ms/batch, {stats['e2e_img_s']*1000:.4f} ms/img, {stats['e2e_fps']:.1f} FPS"
)
