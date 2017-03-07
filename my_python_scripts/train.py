#!/usr/bin/env python
import sys
import caffe

solver = caffe.get_solver('D:\Szakdolgozat\caffe-windows\examples\cifar10\cifar10_quick_solver.prototxt')
#solver.solve()
test_iters = 500

for i in range(test_iters):
    print i
    solver.net.backward()
    solver.step(i)

