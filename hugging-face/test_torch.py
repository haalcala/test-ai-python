import torch as T

print(T.cuda.is_available())

cuda0 = T.device('cuda:0')
print("cuda0:", cuda0)

print("T.tensor([1., 2.], device=cuda0):", T.tensor([1., 2.], device=cuda0))

print("T.cuda.device_count():", T.cuda.device_count())
