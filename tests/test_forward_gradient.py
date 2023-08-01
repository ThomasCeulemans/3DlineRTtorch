##NOTE: this file is based on the pytorch documentation examples
import torch
import torch.autograd.forward_ad as fwAD

primal = torch.randn(4, 4)
tangent = torch.randn(4, 4)

def fn(x, y):
    return x ** 2 / 2.0
    # return x ** 2 + y ** 2

# All forward AD computation must be performed in the context of
# a ``dual_level`` context. All dual tensors created in such a context
# will have their tangents destroyed upon exit. This is to ensure that
# if the output or intermediate results of this computation are reused
# in a future forward AD computation, their tangents (which are associated
# with this computation) won't be confused with tangents from the later
# computation.
with fwAD.dual_level():
    # To create a dual tensor we associate a tensor, which we call the
    # primal with another tensor of the same size, which we call the tangent.
    # If the layout of the tangent is different from that of the primal,
    # The values of the tangent are copied into a new tensor with the same
    # metadata as the primal. Otherwise, the tangent itself is used as-is.
    #
    # It is also important to note that the dual tensor created by
    # ``make_dual`` is a view of the primal.
    dual_input = fwAD.make_dual(primal, tangent)
    assert fwAD.unpack_dual(dual_input).tangent is tangent

    # To demonstrate the case where the copy of the tangent happens,
    # we pass in a tangent with a layout different from that of the primal
    dual_input_alt = fwAD.make_dual(primal, tangent.T)
    assert fwAD.unpack_dual(dual_input_alt).tangent is not tangent

    # Tensors that do not have an associated tangent are automatically
    # considered to have a zero-filled tangent of the same shape.
    plain_tensor = torch.randn(4, 4)
    print("plain tensor: ", plain_tensor)
    print("dual input: ", dual_input)
    dual_output = fn(dual_input, plain_tensor)

    print("dual output (look at tangent):", dual_output)
    print("actual solution: ", primal*tangent)

    # Unpacking the dual returns a ``namedtuple`` with ``primal`` and ``tangent``
    # as attributes
    jvp = fwAD.unpack_dual(dual_output).tangent

assert fwAD.unpack_dual(dual_output).tangent is None