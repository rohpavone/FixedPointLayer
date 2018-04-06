# FixedPointLayer
pyCaffe layer that emulates an arbitrarily sized fixed point activation by restricting its input.

Input: 32-bit floating-point number that you wish to treat like a fixed point number of arbitrary size.

To use:
* Add layer to prototxt (this is a python layer. See how to add here: https://github.com/rohpavone/FixedPointLayer.git)
  * top should be set to input of the next layer (originally fed by the input to the new layer)
  * bottom should be the previous layer you wish to treat as a fixed point
* param_str should have three parameters: `param_str: '{ "mantissa_bits": 8, "min_exp": 131, "max_exp": 96 }'`

An example layer usage would be:
```
layer {
  name: "foo"
  type: "Python"
  top: "foo_OUT"

  bottom: "foo_IN"   #let's suppose we have these two bottom blobs

  python_param {
    module: "FixedPointLayer"
    layer: "FixedPointLayer"
    param_str: '{ "mantissa_bits": 8, "min_exp": 90, "max_exp": 96 }'
  }
}
```
The parameters are:
* `mantissa_bits` : the number of bits (MSB) that are preserved in the floating point (and used in the fixed point)
* `min_exp` : the minimum acceptable exponent (as absolute value) - if an incoming input has an exponent below this, it will be set to 0.
* `max_exp` : the maximum acceptable exponent (as absolute value) - if an incoming input has an exponent above this, it will have its exponent set to this value, and its mantissa set entirely to 1s (maximum possible value)
