import torch.nn as nn
import torch.nn.functional as F
import torch
import functools


def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, output_padding=0,
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, output_padding=output_padding,stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    
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
    #alist = torch.tensor(data_loader.gen_rand_cond_for_binary_au(au.cpu().detach().numpy())).to(device).type(dtype)
    ##
    z = g.encode(x_real)
    ##
    fakes_1 = g.decode(z,des_au_1)
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
    #d_cl_loss = F.l1_loss(d_reg_true, au)
    d_cl_loss = F.binary_cross_entropy_with_logits(d_reg_true, au)
    d_adv_loss = -d_adv_logits_true.mean() + d_adv_logits_fake.mean()
    d_loss = d_adv_loss+lambda_cl*d_cl_loss+10*((grad_norm - 1)**2).mean()
    d_loss_dict = {'d_adv_loss': d_adv_loss , "d_cl_loss": d_cl_loss}
    ## opt. discr. 
    d_optimizer.zero_grad()
    d_loss.backward(retain_graph=True)
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
    #alist = torch.tensor(data_loader.gen_rand_cond_for_binary_au(au.cpu().detach().numpy())).to(device).type(dtype)
    ##
    z = g.encode(x_real)
    ##
    fakes_1 = g.decode(z,des_au_1)
    #fakes_1 = g.translate_decode(z,des_au_1)
    img_rec = g.decode(z,au)
    #
    #d_adv_logits_true, d_reg_true = d(x_real)
    d_adv_logits_fake, d_reg_fake = d(fakes_1)
    #
    g_adv_loss = -d_adv_logits_fake.mean()
    #g_cl_loss = F.l1_loss(d_reg_fake, des_au_1)
    g_cl_loss = F.binary_cross_entropy_with_logits(d_reg_fake, des_au_1)
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

MAX_DIM = 64 * 16  # 1024
class Generator(nn.Module):
    def __init__(self, enc_dim=64, enc_layers=5, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=5, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=13, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)
        self.inject_layers = min(inject_layers, dec_layers - 1)
        self.f_size = img_size // 2**enc_layers  # f_size = 4 for 128x128
        
        layers = []
        n_in = 3
        for i in range(enc_layers):
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn)]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
    
        layers = []
        n_in = n_in + n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if self.shortcut_layers > i:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                        n_in, n_out, (4, 4), stride=2, padding=1, output_padding=1,norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            elif i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-1), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                        n_in, n_out, (4, 4), stride=2, padding=1, output_padding=0,norm_fn=dec_norm_fn, acti_fn=dec_acti_fn)]
                n_in = n_out
                n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh' )]
        self.dec_layers = nn.ModuleList(layers)
        
       
    def encode(self, x):
        z = x
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            zs.append(z)
        return zs
    
    def decode(self, zs, a):
        a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        z = torch.cat([zs[-1], a_tile], dim=1)
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            if self.shortcut_layers > i:  # Concat 1024 with 512
                #print("z",z.shape)
                #print("zs[len(self.dec_layers) - 2 - i]",zs[len(self.dec_layers) - 2 - i].shape)
                z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
            if self.inject_layers > i:
                a_tile = a.view(a.size(0), -1, 1, 1) \
                          .repeat(1, 1, self.f_size * 2**(i+1), self.f_size * 2**(i+1))
                z = torch.cat([z, a_tile], dim=1)
        return z
    
    def forward(self, x, a=None, mode='enc-dec'):
        if mode == 'enc-dec':
            assert a is not None, 'No given attribute.'
            return self.decode(self.encode(x), a)
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            assert a is not None, 'No given attribute.'
            return self.decode(x, a)
        raise Exception('Unrecognized mode: ' + mode)
        
        
            
class Discriminator(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=5, img_size=128):
        super(Discriminator, self).__init__()
        self.f_size = img_size // 2**n_layers
        
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 17, 'none', 'none')
        )
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc_adv(h), self.fc_cls(h)

if __name__ == '__main__':
    d = Discriminator(
            64, 'instancenorm', 'lrelu',
            1024, 'none', 'relu', 5, 112
        )
    print("******** Discriminator/Classifier ********")
    print(d)
    b = torch.rand(5,3,112,112)
    gan_out , au_reg_out  = d.forward(b)
    print("gan_out::",gan_out.shape)
    print(gan_out)
    print("au_reg_out::",au_reg_out.shape)
    g = Generator(
            64, 5, 'batchnorm', 'lrelu',
            64, 5, 'batchnorm', 'relu',
            17, 1, 0, 112
        )
    print("******** Generator ********")
    print(g)
    b = torch.rand(5,3,112,112)
    au_target = torch.rand(5,17)
    a = g(b,au_target)
    print("a",a.shape)
