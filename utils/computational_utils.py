
def compute_input_gradient(func, inp, **kwargs):
	inp.requires_grad = True
	loss = func(inp, **kwargs)
	loss.backward()
	inp.requires_grad = False
	
	return inp.grad.data
