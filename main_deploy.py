import argparse


#def define_argparser():
    # p = argparse.ArgumentParser()
    #
    # p.add_argument("--model_fn", required=False)
    # p.add_argument("--gpu_id", type=int, default=0 if torch.cuda.is_available() else -1)
    #
    # p.add_argument("--train_ratio", type=float, default=.8)
    #
    # p.add_argument("--batch_size", type=int, default=256)
    # p.add_argument("--n_epochs", type=int, default=10)
    #
    # p.add_argument("--model", default="fc", choices=["fc", "cnn"])
    #
    # p.add_argument("--n_layers", type=int, default=5)
    # p.add_argument("--use_dropout", action="store_true")
    # p.add_argument("--dropout_p", type=float, default=.3)
    #
    # p.add_argument("--verbose", type=int, default=1)
    #
    # config = p.parse_args(args=[])
    # config.model_fn='/content/model3.pth'


if __name__ == "__main__":
    #CFG = {
    #    'IMG_SIZE': 224,
    #    'SEED': 41,
    #    'num_worker': multiprocessing.cpu_count(),
    #}


    parser = argparse.ArgumentParser()
    print(parser)
    # We pass in a explicit notebook arg so that we can provide an ordered list
    # and produce an ordered PDF.
    #parser.add_argument("--notebooks", type=str, nargs="+", required=True)
    #parser.add_argument("--pdf_filename", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    args.seed = 41
    args.img_size = 224

    print(vars(args))
    #print(args.epochs)
