#ifdef USE_NNPACK
#include <cerrno>
#include <cstdlib>

#include "nnpack.h"

#include "caffe/layers/nnpack_convolution_layer.hpp"

namespace caffe {

namespace {

nnp_convolution_algorithm nnp_algorithm(NNPACKConvolutionParameter_Algorithm algo) {
  nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;
  switch (algo) {
    case NNPACKConvolutionParameter_Algorithm_AUTO:
      algorithm = nnp_convolution_algorithm_auto;
      break;
    case NNPACKConvolutionParameter_Algorithm_WINOGRAD: {
      algorithm = nnp_convolution_algorithm_wt8x8;
      break;
    }
    case NNPACKConvolutionParameter_Algorithm_FFT_16x16: {
      algorithm = nnp_convolution_algorithm_ft16x16;
      break;
    }
    case NNPACKConvolutionParameter_Algorithm_FFT_8x8: {
      algorithm = nnp_convolution_algorithm_ft8x8;
      break;
    }
  }
  return algorithm;
}

nnp_convolution_transform_strategy nnp_kts(
    NNPACKConvolutionParameter_KernelTransformStrategy kts) {
  nnp_convolution_transform_strategy kernel_transform_strategy =
      nnp_convolution_transform_strategy_compute;
  switch (kts) {
    case NNPACKConvolutionParameter_KernelTransformStrategy_RECOMPUTE:
      kernel_transform_strategy =
          nnp_convolution_transform_strategy_compute;
      break;
    case NNPACKConvolutionParameter_KernelTransformStrategy_REUSE: {
      kernel_transform_strategy =
          nnp_convolution_transform_strategy_reuse;
      break;
    }
  }
  return kernel_transform_strategy;
}

void caffe_nnp_convolution_forward(
    NNPACKConvolutionParameter_Algorithm algo,
    NNPACKConvolutionParameter_KernelTransformStrategy kts,
    const Blob<float>& bottom,
    const Blob<float>& weights,
    const Blob<float>& bias,
    const Blob<int>& pad,
    Blob<float>* top) {
  VLOG(1) << "NNPack Convolution Algo:"
          << NNPACKConvolutionParameter_Algorithm_Name(algo);
  VLOG(1) << "NNPack Convolution KTS:"
          << NNPACKConvolutionParameter_KernelTransformStrategy_Name(kts);
  const size_t batch_size = bottom.num();
  const size_t input_channels = bottom.channels();
  const size_t output_channels = top->channels();
  const nnp_size input_size = {static_cast<size_t>(bottom.width()),
                               static_cast<size_t>(bottom.height())};
  const nnp_size kernel_size = {
      .width = static_cast<size_t>(weights.width()),
      .height = static_cast<size_t>(weights.height())};
  const nnp_size output_subsampling = { 1, 1 };
  const nnp_padding padding = {.top = static_cast<size_t>(pad.cpu_data()[0]),
                               .right = static_cast<size_t>(pad.cpu_data()[1]),
                               .bottom = static_cast<size_t>(pad.cpu_data()[0]),
                               .left = static_cast<size_t>(pad.cpu_data()[1])};

  const nnp_convolution_algorithm algorithm = nnp_algorithm(algo);
  const nnp_convolution_transform_strategy kernel_transform_strategy = nnp_kts(kts);

  if (batch_size == 1) {
    VLOG(1) << "Running inference mode";
    const nnp_status status = nnp_convolution_inference(
        algorithm,
        kernel_transform_strategy,
        input_channels,
        output_channels,
        input_size,
        padding,
        kernel_size,
        output_subsampling,
        bottom.cpu_data(),
        weights.cpu_data(),
        bias.cpu_data(),
        top->mutable_cpu_data(),
        Caffe::nnpack_threadpool(),
        NULL);
    CHECK_EQ(nnp_status_success, status);
  } else {
    VLOG(1) << "Running batched mode";
    const nnp_status status = nnp_convolution_output(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        padding,
        kernel_size,
        bottom.cpu_data(),
        weights.cpu_data(),
        bias.cpu_data(),
        top->mutable_cpu_data(),
        Caffe::nnpack_threadpool(),
        NULL);
    CHECK_EQ(nnp_status_success, status);
  }
}

}


template <typename Dtype>
void NNPackConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  return ConvolutionLayer<Dtype>::Forward_cpu(bottom, top);
}

template <>
void NNPackConvolutionLayer<float>::Forward_cpu(
    const vector<Blob<float>*>& bottom,
    const vector<Blob<float>*>& top) {
  if (!this->bias_term_) {
    VLOG(1) << "NNPACK Convolution requires a bias term, falling back";
    return ConvolutionLayer<float>::Forward_cpu(bottom, top);
  }

  bool is_stride_1 = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (this->stride_.cpu_data()[i] != 1) {
      is_stride_1 = false;
    }
  }

  if (!is_stride_1) {
    VLOG(1) << "NNPACK Convolution requires strdie 1, falling back";
    return ConvolutionLayer<float>::Forward_cpu(bottom, top);
  }

  CHECK(this->bias_term_);
  CHECK(is_stride_1);
  for (int i = 0; i < bottom.size(); ++i) {
    caffe_nnp_convolution_forward(
        this->layer_param_.nnpack_convolution_param().algorithm(),
        this->layer_param_.nnpack_convolution_param()
            .kernel_transform_strategy(),
        *(bottom[i]),
        *(this->blobs_[0]),
        *(this->blobs_[1]),
        this->pad_,
        top[i]);
  }
}

template <typename Dtype>
void NNPackConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(ERROR) << "Not implemented";
}

INSTANTIATE_CLASS(NNPackConvolutionLayer);

} // namespace caffe
#endif
