from torch.utils.data import DataLoader

from data_provider.data_loader import data


def data_provider(args, flag):
    Data = data
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len]
    )

    data_loader =DataLoader(
        data_set,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        #num_workers=args.num_workers,
        batch_size=batch_size
    )
    return data_set, data_loader
