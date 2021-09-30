import torch



num_classes = 1

############################ FCN
# from model.fcn.fcn import FCN16s
# model = FCN16s(nclass=num_classes)
# x = torch.randn(2, 3, 512, 512)
# out = model(x)  # tuple()
# # import ipdb; ipdb.set_trace()
# print(out[0].shape)


from model.pspnet.pspnet import get_psp_resnet50_voc
model = get_psp_resnet50_voc()
img = torch.randn(4, 3, 512, 512)
output = model(img)
import ipdb; ipdb.set_trace()