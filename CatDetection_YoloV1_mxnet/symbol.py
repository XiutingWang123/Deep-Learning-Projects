import mxnet as mx


def YOLO_loss(predict, label):
    """
    :param predict: size(batch_size, 7, 7, 5) NDArray
    :param label: size(batch_size, 7, 7, 5) NDArray
    :return: loss, size(batch_size, 49, 1) NDArray
    """

    # reshape input to desired shape
    predict = mx.sym.reshape(predict, shape=(-1, 49, 5))

    # shift values from softsign (-1, 1) to (0, 1). softsign = x / (1 + |x|)
    predict_shift = (predict + 1) / 2
    label = mx.sym.reshape(label, shape=(-1, 49, 5))

    # split the tensor in the order of [prob, x, y, w, h]
    cl, xl, yl, wl, hl = mx.sym.split(label, num_outputs=5, axis=2)
    cp, xp, yp, wp, hp = mx.sym.split(predict_shift, num_outputs=5, axis=2)

    # increase loss for bounding box containing cat, decrease loss otherwise
    lambda_coord = 5
    lambda_obj = 1
    lambda_noobj = 0.5
    mask = cl * lambda_obj + (1 - cl) * lambda_noobj

    # compute loss
    lossc = mx.sym.LinearRegressionOutput(label=cl*mask, data=cp*mask)
    lossx = mx.sym.LinearRegressionOutput(label=xl*cl*lambda_coord, data=xp*cl*lambda_coord)
    lossy = mx.sym.LinearRegressionOutput(label=yl*cl*lambda_coord, data=yp*cl*lambda_coord)
    lossw = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(wl)*cl*lambda_coord, data=mx.sym.sqrt(wp)*cl*lambda_coord)
    lossh = mx.sym.LinearRegressionOutput(label=mx.sym.sqrt(hl)*cl*lambda_coord, data=mx.sym.sqrt(hp)*cl*lambda_coord)

    #joint loss
    loss = lossc + lossx + lossy + lossw + lossh

    return loss


def get_resnet_model(model_path, epoch):
    """
    get pretrained imagnet model
    :param model_path: path of pretrained model
    :param epoch: number of epoch of given model
    :return: a group symbol
    """

    label = mx.sym.Variable('softmax_label')

    # load symbol and actual weights
    sym, args, aux = mx.model.load_checkpoint(model_path, epoch)
    # extract last BN layer and previous layers
    sym = sym.get_internals()['bn1_output']
    # append two layers
    # the resnet has done 5 times down scale (2*2 stride)
    # original size 224 / 2^5 = 7, thus we have 7 * 7 output
    sym = mx.sym.Activation(data=sym, act_type='relu')
    sym = mx.sym.Convolution(data=sym, kernel=(3,3), num_filter=5, pad=(1,1), stride=(1,1), no_bias=True)

    # get softsign
    sym = sym / (1 + mx.sym.abs(sym))
    # transpose from (number of img in a batch, 5, 7, 7) to (number of img, 7, 7, 5)
    logit = mx.sym.transpose(sym, axes=(0, 2, 3, 1), name='logit')

    # apply loss
    loss_ = YOLO_loss(logit, label)
    loss = mx.sym.MakeLoss(loss_)
    # multi-output logit should be blocked from generating gradients
    out = mx.sym.Group([mx.sym.BlockGrad(logit), loss])

    return out

