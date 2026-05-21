#include <torch/extension.h>

void cfhead_coarse_forward_cuda(torch::Tensor lut, torch::Tensor index_t, torch::Tensor out);
void cfhead_fine_forward_cuda(torch::Tensor hidden, torch::Tensor weight, torch::Tensor candidate_idx, torch::Tensor out);

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor coarse_forward(torch::Tensor lut, torch::Tensor index_t) {
    CHECK_INPUT(lut);
    CHECK_INPUT(index_t);
    TORCH_CHECK(lut.dim() == 3, "lut must have shape [B, num_tiles, cluster_size]");
    TORCH_CHECK(index_t.dim() == 2, "index_t must have shape [num_tiles, vocab_size]");
    TORCH_CHECK(lut.scalar_type() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(index_t.scalar_type() == torch::kUInt8, "index_t must be uint8");
    TORCH_CHECK(lut.size(1) == index_t.size(0), "lut/index_t num_tiles mismatch");

    auto out = torch::empty({lut.size(0), index_t.size(1)}, lut.options().dtype(torch::kFloat32));
    cfhead_coarse_forward_cuda(lut, index_t, out);
    return out;
}

torch::Tensor fine_forward(torch::Tensor hidden, torch::Tensor weight, torch::Tensor candidate_idx) {
    CHECK_INPUT(hidden);
    CHECK_INPUT(weight);
    CHECK_INPUT(candidate_idx);
    TORCH_CHECK(hidden.dim() == 2, "hidden must have shape [B, H]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [V, H]");
    TORCH_CHECK(candidate_idx.dim() == 2, "candidate_idx must have shape [B, K]");
    TORCH_CHECK(candidate_idx.scalar_type() == torch::kInt64, "candidate_idx must be int64");
    TORCH_CHECK(hidden.scalar_type() == weight.scalar_type(), "hidden/weight dtype mismatch");
    TORCH_CHECK(hidden.size(0) == candidate_idx.size(0), "hidden/candidate batch mismatch");
    TORCH_CHECK(hidden.size(1) == weight.size(1), "hidden/weight hidden-size mismatch");

    auto out = torch::empty({candidate_idx.size(0), candidate_idx.size(1)}, hidden.options().dtype(torch::kFloat32));
    cfhead_fine_forward_cuda(hidden, weight, candidate_idx, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coarse_forward", &coarse_forward, "CFHead coarse forward (CUDA)");
    m.def("fine_forward", &fine_forward, "CFHead fine candidate logits (CUDA)");
}
