import caffe
import numpy as np

def determine_mask(limit, truncate, exp=False):
    if not exp:
        return ((1 << truncate) - 1) << (limit - truncate)
    else:
        # truncate from the top
        return (1 << truncate) - 1


class FixedPointLayer(caffe.Layer):
    def quantize_exp(self, mant_trunc,bottom, top, limit):
        # track negative indices
        negative = bottom[0].data.copy()
        negative[negative < 0] = -1.0
        negative[negative >=0] = 1.0
        
        # get  32bit representation
        n = np.fabs(bottom[0].data).view(np.int32)
        
        # exponents
        exp = np.right_shift(n, limit)
    
        # mantissa
        mantissa = np.bitwise_and(n, np.int32(2**(limit + 1) - 1))
        mantissa[exp > self.max_exponent]  = self.max_mantissa
        mantissa[exp < self.min_exponent] = self.min_mantissa
        
        exp[exp > self.max_exponent] = self.max_exponent
        exp[exp < self.min_exponent] = 0
            
        # truncate each of these arrays
        mantissa = np.bitwise_and(mantissa, mant_trunc)
        
        top[0].data[...] = np.multiply(np.bitwise_or(np.left_shift(exp, limit), mantissa).view(np.float32), negative)


    def setup(self, bottom, top):
        # have defined:
        #    number of levels
        params = eval(self.param_str)
        if len(bottom) != 1:
            raise Exception("Can only have one input")

        if len(top) != 1:
            raise Exception("Can only have one output")

        self.mantissa = params['mantissa_bits']
        self.min_exponent = params['min_exp']
        self.max_exponent = params['max_exp']

        top[0].reshape(*bottom[0].data.shape)

        self.trunc_mantissa = determine_mask(23, self.mantissa)
        self.max_mantissa = self.trunc_mantissa
        self.min_mantissa = 0x0

    def reshape(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

    def forward(self, bottom, top):
        self.quantize_exp(self.trunc_mantissa, bottom, top, 23) 

if __name__ == '__main__':
    top = []
    bottom = []
