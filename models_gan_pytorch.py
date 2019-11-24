import torch.nn as nn
import torch.nn.functional as F
import torch
import functools


class ModuleBase(nn.Module):
    def __init__(self):
        super(ModuleBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'weight'):
                torch.nn.init.xavier_uniform(m.weight)
                #m.weight.data.normal_(0.0, 0.02)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.01)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer
    
    
def train_D_wasserstein_gp(g, d, x_real, au, lambda_cl, lambda_cyc, data_loader,device,d_optimizer):
    for p in d.parameters():
        p.requires_grad = True
    batch_size = x_real.shape[0]
    #
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
    des_au_1 = torch.tensor(data_loader.gen_rand_cond(batch_size=batch_size)).to(device).type(dtype)
    ##
    z = g.encode(x_real)
    ##
    fakes_1 = g.translate_decode(z,des_au_1)
    #img_rec = g.translate_decode(z,au)
    #
    d_adv_logits_true, d_reg_true = d(x_real)
    d_adv_logits_fake, d_reg_fake = d(fakes_1)
    #
    alpha = torch.rand(batch_size,1,1,1,device=device)
    x_gp = alpha*fakes_1+(1-alpha)*x_real
    d_gp,_ = d(x_gp)
    grad = torch.autograd.grad(d_gp.sum(), x_gp, create_graph=True)
    grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)
    #
    d_cl_loss = F.l1_loss(d_reg_true, au)
    d_adv_loss = -d_adv_logits_true.mean() + d_adv_logits_fake.mean()
    d_loss = d_adv_loss+lambda_cl*d_cl_loss+10*((grad_norm - 1)**2).mean()
    d_loss_dict = {'d_adv_loss': d_adv_loss , "d_cl_loss": d_cl_loss}
    ## opt. discr. 
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    #
    return d_loss_dict
    
def train_G_wasserstein_gp(g, d, x_real, au, lambda_cl, lambda_cyc, data_loader,device,g_optimizer):
    for p in d.parameters():
        p.requires_grad = False
    batch_size = x_real.shape[0]
    #
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
    des_au_1 = torch.tensor(data_loader.gen_rand_cond(batch_size=batch_size)).to(device).type(dtype)
    ##
    z = g.encode(x_real)
    ##
    fakes_1 = g.translate_decode(z,des_au_1)
    img_rec = g.translate_decode(z,au)
    #
    #d_adv_logits_true, d_reg_true = d(x_real)
    d_adv_logits_fake, d_reg_fake = d(fakes_1)
    #
    g_adv_loss = -d_adv_logits_fake.mean()
    g_cl_loss = F.l1_loss(d_reg_fake, des_au_1)
    rec_loss = F.l1_loss(img_rec, x_real)
    g_loss = g_adv_loss + lambda_cl*g_cl_loss + lambda_cyc*rec_loss
    g_loss_dict = {'g_adv_loss': g_adv_loss , "g_cl_loss": g_cl_loss, "rec_loss": rec_loss}
    #
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    #
    return g_loss_dict
    
    
def loss_nonsaturating(g, d, x_real, au, lambda_cl, lambda_cyc, data_loader,device,train_generator=True):
    batch_size = x_real.shape[0]
    ## TODO repeat 
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor 
    des_au_1 = torch.tensor(data_loader.gen_rand_cond(batch_size=batch_size)).to(device).type(dtype)
    ##
    z = g.encode(x_real)
    ##
    fakes_1 = g.translate_decode(z,des_au_1)
    img_rec = g.translate_decode(z,au)
    ## d adv loss 
    d_adv_logits_true, d_reg_true = d(x_real)
    d_adv_logits_fake, d_reg_fake = d(fakes_1)
    d_adv_loss = -F.logsigmoid(d_adv_logits_true).mean() - F.logsigmoid(-d_adv_logits_fake).mean()
    ## d cond reg. loss 
    d_cl_loss = F.l1_loss(d_reg_true, au)
    
    ## d_loss 
    d_loss = d_adv_loss + lambda_cl*d_cl_loss
    d_loss_dict = {'d_adv_loss': d_adv_loss , "d_cl_loss": d_cl_loss}
    
    ## g_loss 
    if train_generator:
        g_adv_loss = -F.logsigmoid(d_adv_logits_fake).mean()
        g_cl_loss = F.l1_loss(d_reg_fake, des_au_1)
        ## rec. loss 
        rec_loss = F.l1_loss(img_rec, x_real)
        g_loss = g_adv_loss + lambda_cl*g_cl_loss + lambda_cyc*rec_loss
        g_loss_dict = {'g_adv_loss': g_adv_loss , "g_cl_loss": g_cl_loss, "rec_loss": rec_loss}
        ## 
        return d_loss , d_loss_dict , g_loss, g_loss_dict
    else:
        return d_loss, d_loss_dict, None, None 

class Generator(ModuleBase):
    def __init__(self, img_shape,gf,AU_num,num_layers=4,f_size=6,tranform_layer=False,res_blocks=1):
        super(Generator, self).__init__()
        
        channels, height, width = img_shape
        
        self.res_blocks = res_blocks
        
        
        class Conv2dBlock(ModuleBase):
            def __init__(self, in_filters, out_filters, f_size=6, normalize=True,stride=2,padding=0):
                super(Conv2dBlock, self).__init__()
                layers = [nn.Conv2d(in_filters, out_filters, f_size, stride=stride, padding=padding)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters,affine=True))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x)
        
        def conv2d(in_filters, out_filters, f_size=6, normalize=True,stride=2,padding=0):
            return Conv2dBlock(in_filters, out_filters, f_size, normalize,stride,padding)
        
        class Deconv2dBlock(ModuleBase):
            def __init__(self, in_filters, out_filters, f_size=6, stride=2,padding=0,normalize=True,bias=True,output_padding=0):
                super(Deconv2dBlock, self).__init__()
                layers = [nn.ConvTranspose2d(in_filters, out_filters, kernel_size=f_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]
                if normalize:
                    layers.append(nn.InstanceNorm2d(out_filters, affine=True))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                self.layers = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.layers(x)
        
        def deconv2d(in_filters, out_filters, f_size=6, stride=2,padding=0,normalize=True,bias=True,output_padding=0):
             return Deconv2dBlock(in_filters, out_filters, f_size, stride,padding,normalize,bias,output_padding)
            
        ### ENCODER
        layers = [
            conv2d(channels, gf, f_size=6,stride=3),
            conv2d(gf, gf*2,f_size=6),
            conv2d(gf*2, gf*4,f_size=6),
            conv2d(gf*4, gf*8,f_size=6)
        ]
        self.enc_layers = nn.ModuleList(layers)
        
        ### 1x1 conv 
        self.conv_1_1 = nn.Conv2d(gf*8+AU_num, gf*8, kernel_size=1, stride=1, padding=0)
        
        ### DECODER
        layers = [
            deconv2d(gf*8, gf*4, f_size=6), ## residual block
            deconv2d(gf*8, gf*2, f_size=6), 
            deconv2d(gf*2, gf, f_size=6),    
            deconv2d(gf, channels,f_size=6,stride=3,output_padding=1)
        ]
        self.dec_layers = nn.ModuleList(layers)
        
       
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def translate_decode(self, z,au_target):
        _au_target = au_target.view(au_target.size(0), -1, 1, 1)
        _z = z[-1]
        #print("_au_target",_au_target.shape)
        #print("_z",_z.shape,len(z))
        res_block = torch.cat([_z, _au_target], dim=1)
        res_block = self.conv_1_1(res_block)
        for i,layer in enumerate(self.dec_layers):
            res_block = layer(res_block)
            if len(z) > 0 and i < self.res_blocks:
                _z = z[-2-i]
                #print("res_block",res_block.shape)
                #print("_z",_z.shape,len(z))
                res_block = torch.cat([res_block, _z], dim=1)
        return res_block
    
    def forward(self, img,au_target):
        return self.translate_decode(self.encode(img),au_target)
        
        
            
class Discriminator(ModuleBase):
    def __init__(self, img_shape,df,AU_num):
        super(Discriminator, self).__init__()

        channels, height, width = img_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape_PatchGAN = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, f_size=4, normalize=True,stride=2):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, f_size, stride=stride, padding=0)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *discriminator_block(channels, df, normalize=False,f_size=6,stride=3),
            *discriminator_block(64, 128,f_size=6),
            *discriminator_block(128, 256,f_size=6),
            *discriminator_block(256, 512,f_size=6),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            ##nn.Conv2d(512, 1, 4, padding=1)
        )
        self.gan_task = nn.Sequential(
            torch.nn.Linear(512,512),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            torch.nn.Linear(512,1)
        )
        self.au_reg_task = nn.Sequential(
            torch.nn.Linear(512,512),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(),
            torch.nn.Linear(512,AU_num)
        )

    def forward(self, img):
        #print("img:",img.shape)
        common_pass = self.net.forward(img).view(img.shape[0],-1)
        #print("common_pass::",common_pass.shape)
        gan_out = self.gan_task(common_pass)
        au_reg_out = self.au_reg_task(common_pass)
        return gan_out , au_reg_out

if __name__ == '__main__':
    d = Discriminator(img_shape=(3,112,112),df=64,AU_num=17)
    print("******** Discriminator/Classifier ********")
    print(d)
    b = torch.rand(5,3,112,112)
    gan_out , au_reg_out  = d.forward(b)
    print("gan_out::",gan_out.shape)
    print("au_reg_out::",au_reg_out.shape)
    g = Generator(img_shape=(3,112,112),gf=64,AU_num=17)
    print("******** Generator ********")
    print(g)
    b = torch.rand(5,3,112,112)
    au_target = torch.rand(5,17)
    a = g.forward(b,au_target)
    print("a",a.shape)
