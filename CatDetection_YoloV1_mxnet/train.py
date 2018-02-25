import mxnet as mx
import numpy as np
from symbol import get_resnet_model
from data_ulti import get_iterator

import logging
import sys
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

if __name__ == "__main__":

    # get sym
    sym = get_resnet_model('pretrained_models/resnet-18', 0)
    _, args_params, aux_params = mx.model.load_checkpoint('pretrained_models/resnet-18', 0)

    # get data
    train_data = get_iterator(path='DATA_rec/cat.rec', data_shape=(3, 224, 224), label_width=7*7*5, batch_size=32, shuffle=True)
    val_data = get_iterator(path='DATA_rec/cat_val.rec', data_shape=(3,224,224), label_width=7*7*5, batch_size=32)

    mod = mx.mod.Module(symbol=sym, context=mx.cpu(0))

    def loss_metric(label, pred):
        label = label.reshape((-1, 7, 7, 5))
        pred = pred.reshape((-1, 7, 7, 5))
        pred_shift = (pred + 1) / 2
        cl = label[:, :, :, 0]
        xl = label[:, :, :, 1] * 32
        yl = label[:, :, :, 2] * 32
        wl = label[:, :, :, 3] * 224
        hl = label[:, :, :, 4] * 224
        cp = pred_shift[:, :, :, 0]
        xp = pred_shift[:, :, :, 1] * 32
        yp = pred_shift[:, :, :, 2] * 32
        wp = pred_shift[:, :, :, 3] * 224
        hp = pred_shift[:, :, :, 4] * 224

        num_box = np.sum(cl)
        FN = np.sum(cl * (cp < 0.5) == 1)
        FP = np.sum((1 - cl) * (cp > 0.5))
        print "Recall is {}".format(np.sum(cp[cl == 1.0] > 0.5)/np.sum(cl))
        print "precision is {}".format(np.sum(cp[cl == 1.0] > 0.5)/np.sum(cp > 0.5)+2e-5)
        print "Number of FN is {}".format(FN)
        print "Number of FP is {}".format(FP)
        print "The total number of boxes is {}".format(num_box)
        print "FN boxes: {}".format(np.where(cl*(cp < 0.5) == 1))
        print "Mean difference of prob : {}".format((np.mean(np.abs(cl-cp))))
        print "Mean difference of TP boxes prob: {}".format(np.mean(np.abs(cl[cl == 1.] - cp[cl == 1.])))
        print "Mean difference of TN boxes prob: {}".format(np.mean(np.abs(cl[cl != 1.] - cp[cl != 1.])))
        print "Mean difference of x: {}".format(np.mean(np.abs(xl[cl == 1] - xp[cl == 1])))
        print "Mean difference of y: {}".format(np.mean(np.abs(yl[cl == 1] - yp[cl == 1])))
        print "Mean difference of w: {}".format(np.mean(np.abs(wl[cl == 1] - wp[cl == 1])))
        print "Mean difference of h: {}".format(np.mean(np.abs(hl[cl == 1] - hp[cl == 1])))

        return -1


    # setup metric
    metric = mx.metric.create(loss_metric, allow_extra_outputs=True)

    # setup monitor for debugging
    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)
    mon = None # mx.mon.Monitor(10, norm_stat, pattern=".*backward*.")

    # save model
    checkpoint = mx.callback.do_checkpoint('cat_model')

    #train
    mod.fit(train_data=train_data,
            eval_data=val_data,
            num_epoch=500,
            monitor=mon,
            eval_metric=[metric],
            optimizer='rmsprop',
            optimizer_params={'learning_rate':0.01, 'lr_scheduler':mx.lr_scheduler.FactorScheduler(300000, 0.1, 0.001)},
            allow_missing=True,
            initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
            arg_params=args_params, aux_params=aux_params,
            batch_end_callback=[mx.callback.Speedometer(batch_size=32, frequent=10, auto_reset=False)],
            epoch_end_callback=checkpoint,)



