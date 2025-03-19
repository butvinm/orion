def test_orion_core_imports():
    """Test that core Orion modules can be imported."""
    import orion
    import orion.nn as on
    import orion.models as models
    import orion.core.utils as utils
    
    assert orion.__name__ == "orion"
    assert on.__name__ == "orion.nn"
    assert models.__name__ == "orion.models"
    assert utils.__name__ == "orion.core.utils"

def test_linear_transforms():
    """Test that linear transform modules can be instantiated."""
    import orion.nn as on
    
    linear = on.Linear(10, 10)
    assert isinstance(linear, on.Linear)
    
    conv = on.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
    assert isinstance(conv, on.Conv2d)
    
    avg_pool = on.AvgPool2d(kernel_size=3, stride=3)
    assert isinstance(avg_pool, on.AvgPool2d)
    
    adaptive_pool = on.AdaptiveAvgPool2d(output_size=1)
    assert isinstance(adaptive_pool, on.AdaptiveAvgPool2d)

def test_activation_functions():
    """Test that activation function modules can be instantiated."""
    import orion.nn as on
    
    activation = on.Activation(coeffs=[1,0,0])
    assert isinstance(activation, on.Activation)
    
    quad = on.Quad()
    assert isinstance(quad, on.Quad)
    
    sigmoid = on.Sigmoid(degree=31)
    assert isinstance(sigmoid, on.Sigmoid)
    
    silu = on.SiLU(degree=127)
    assert isinstance(silu, on.SiLU)
    
    gelu = on.GELU()
    assert isinstance(gelu, on.GELU)
    
    relu = on.ReLU(degrees=[15,15,27], logalpha=6, logerr=12)
    assert isinstance(relu, on.ReLU)

def test_normalization():
    """Test that normalization modules can be instantiated."""
    import orion.nn as on
    
    bn1d = on.BatchNorm1d(32)
    assert isinstance(bn1d, on.BatchNorm1d)
    
    bn2d = on.BatchNorm2d(32)
    assert isinstance(bn2d, on.BatchNorm2d)

def test_operations():
    """Test that operation modules can be instantiated."""
    import orion.nn as on
    
    add = on.Add()
    assert isinstance(add, on.Add)
    
    mult = on.Mult()
    assert isinstance(mult, on.Mult)
    
    bootstrap = on.Bootstrap(-1, 1, input_level=1) # internal module
    assert isinstance(bootstrap, on.Bootstrap)

def test_reshape():
    """Test that reshape modules can be instantiated."""
    import orion.nn as on
    
    flatten = on.Flatten()
    assert isinstance(flatten, on.Flatten)