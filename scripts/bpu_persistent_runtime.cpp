// Persistent F32 boundary adapter for the installed RDK X5 DNN API.
// All tensor shapes/strides are checked; no system configuration is changed.
#include <dnn/hb_dnn.h>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(int code) { if (code) throw std::runtime_error("DNN API error " + std::to_string(code)); }
size_t dense(const hbDNNTensorProperties &p, bool input) {
  if (p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.validShape.numDimensions != 4)
    throw std::runtime_error("Expected four-dimensional F32 boundary");
  if (input && p.tensorLayout != HB_DNN_LAYOUT_NCHW)
    throw std::runtime_error("Expected NCHW input");
  size_t stride = sizeof(float);
  for (int i = 3; i >= 0; --i) {
    if (p.validShape.dimensionSize[i] <= 0 || p.validShape.dimensionSize[i] != p.alignedShape.dimensionSize[i] || p.stride[i] != static_cast<int>(stride))
      throw std::runtime_error("Padded or non-dense boundary requires explicit decoding");
    stride *= p.validShape.dimensionSize[i];
  }
  if (stride > static_cast<size_t>(p.alignedByteSize)) throw std::runtime_error("Invalid allocation size");
  return stride;
}
struct Runtime {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  std::vector<hbDNNTensor> inputs, outputs;
  std::vector<size_t> input_bytes;
  size_t output_bytes = 0;
  ~Runtime() {
    for (auto &t : inputs) if (t.sysMem[0].virAddr) hbSysFreeMem(&t.sysMem[0]);
    for (auto &t : outputs) if (t.sysMem[0].virAddr) hbSysFreeMem(&t.sysMem[0]);
    if (packed) hbDNNRelease(packed);
  }
};
struct Task {
  hbDNNTaskHandle_t handle = nullptr;
  ~Task() { if (handle) hbDNNReleaseTask(handle); }
};
}
extern "C" {
const char *bpu_last_error() { return error.c_str(); }
void *bpu_create(const char *path) {
  auto *r = new Runtime;
  try {
    check(hbDNNInitializeFromFiles(&r->packed, &path, 1));
    const char **names = nullptr; int count = 0;
    check(hbDNNGetModelNameList(&names, &count, r->packed));
    if (count != 1) throw std::runtime_error("Expected exactly one model");
    check(hbDNNGetModelHandle(&r->model, r->packed, names[0]));
    int n = 0; check(hbDNNGetInputCount(&n, r->model));
    r->inputs.resize(n); r->input_bytes.resize(n);
    for (int i = 0; i < n; ++i) {
      auto &t = r->inputs[i];
      check(hbDNNGetInputTensorProperties(&t.properties, r->model, i));
      r->input_bytes[i] = dense(t.properties, true);
      check(hbSysAllocCachedMem(&t.sysMem[0], t.properties.alignedByteSize));
    }
    check(hbDNNGetOutputCount(&n, r->model));
    if (n != 1) throw std::runtime_error("Expected one output");
    r->outputs.resize(n);
    auto &t = r->outputs[0];
    check(hbDNNGetOutputTensorProperties(&t.properties, r->model, 0));
    r->output_bytes = dense(t.properties, false);
    check(hbSysAllocCachedMem(&t.sysMem[0], t.properties.alignedByteSize));
    return r;
  } catch (const std::exception &e) { error = e.what(); delete r; return nullptr; }
}
int bpu_shape(void *handle, int index, int output, int *shape) {
  try {
    auto *r = static_cast<Runtime *>(handle);
    if (!r || index < 0) throw std::runtime_error("Invalid shape request");
    const auto &t = output ? r->outputs.at(index) : r->inputs.at(index);
    std::memcpy(shape, t.properties.validShape.dimensionSize, 4*sizeof(int));
    return 0;
  } catch (const std::exception &e) { error = e.what(); return -1; }
}
int bpu_run(void *handle, const float **data, const size_t *bytes, int count,
            float *output, size_t output_bytes, double *core_ms) {
  try {
    auto *r = static_cast<Runtime *>(handle);
    if (!r || !data || !bytes || !output || count != static_cast<int>(r->inputs.size()) || output_bytes != r->output_bytes)
      throw std::runtime_error("Invalid inference boundary");
    for (int i = 0; i < count; ++i) {
      if (!data[i] || bytes[i] != r->input_bytes[i]) throw std::runtime_error("Input byte size mismatch");
      std::memcpy(r->inputs[i].sysMem[0].virAddr, data[i], bytes[i]);
      check(hbSysFlushMem(&r->inputs[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN));
    }
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    Task task; hbDNNTensor *out = r->outputs.data();
    auto start = std::chrono::steady_clock::now();
    check(hbDNNInfer(&task.handle, &out, r->inputs.data(), r->model, &control));
    check(hbDNNWaitTaskDone(task.handle, 600000));
    if (core_ms) *core_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
    check(hbSysFlushMem(&r->outputs[0].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE));
    std::memcpy(output, r->outputs[0].sysMem[0].virAddr, output_bytes);
    return 0;
  } catch (const std::exception &e) { error = e.what(); return -1; }
}
void bpu_close(void *handle) { delete static_cast<Runtime *>(handle); }
}
