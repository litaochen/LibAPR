//
// Created by Joel Jonsson on 23.07.18.
//

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "../src/wrapper/PyAPR.hpp"
#include "../src/wrapper/PyPixelData.hpp"
#include "../src/wrapper/pythonBind.cpp"


using namespace tensorflow;

REGISTER_OP("APRConv")
.Attr("T: {float, uint16, uint8} = DT_FLOAT")
.Input("inputAPR: PyAPR<T>")
.Output("outputAPR: PyAPR<T>")
.SetShapeFn([](shape_inference::InferenceContext* c) {
c->set_output(0, c->input(0));
return Status::OK();
});

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("APRConv_ds")
    .Attr("T: {float, uint16, uint8} = DT_FLOAT")
    .Input("inputAPR: PyAPR<T>")
    .Input("weights: float")
    .Output("outputAPR: PyAPR<T>")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle weight_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &weight_shape));

        c->set_output(0, c->input(0));
        return Status::OK();
    });

class APRConv_dsOp : public OpKernel {
public:

    // constructor
    explicit APRConv_dsOp(OpKernelConstruction* context) : OpKernel(context) {

    }


    void Compute(OpKernelContext* context) override {

        // make sure the expected number of inputs are present.
        DCHECK_EQ(2, context->num_inputs());

        // get the input tensor
        const Tensor& input = context->input(0);

        // get the weight tensor
        const Tensor& weights = context->input(1);

        // check shapes of input and weights
        const TensorShape& input_shape = input.shape();
        const TensorShape& weights_shape = weights.shape();

        // check input is a standing vector
        DCHECK_EQ(input_shape.dims(), 2);
        DCHECK_EQ(input_shape.dim_size(1), 1);

        // check weights is matrix of correct size
        DCHECK_EQ(weights_shape.dims(), 2);
        DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));

        // create output shape
        TensorShape output_shape;
        output_shape.AddDim(weights_shape.dim_size(0));
        output_shape.AddDim(1);

        // create output tensor
        Tensor* output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        // get the corresponding Eigen tensors for data access
        auto input_tensor = input.matrix<float>();
        auto weights_tensor = weights.matrix<float>();
        auto output_tensor = output->matrix<float>();

        for (int i = 0; i < output->shape().dim_size(0); i++) {
            output_tensor(i, 0) = 0;
            for (int j = 0; j < weights.shape().dim_size(1); j++) {
                output_tensor(i, 0) += weights_tensor(i, j)*input_tensor(j, 0);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("InnerProduct").Device(DEVICE_CPU), InnerProductOp);