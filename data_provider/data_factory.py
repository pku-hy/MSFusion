from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar
from torch.utils.data import DataLoader
data_dict = {'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour,
             'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute,
             'custom': Dataset_Custom, 'Solar': Dataset_Solar}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = args.drop_last
        batch_size = args.batch_size
        freq = args.freq

    
    if args.data == 'm4':
        drop_last = False
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
    )
    if data_set.data_x.shape[-1] != args.c_in:
        raise ValueError(f'Dataset has {data_set.data_x.shape[-1]} channels, but c_in={args.c_in}')
    dataset_len = len(data_set)
    print(flag, dataset_len)
    if dataset_len <= 0:
        raise ValueError(
            f'{flag} split has no valid windows: len={dataset_len}, '
            f'seq_len={args.seq_len}, pred_len={args.pred_len}. '
            'Please check whether the dataset file is complete or reduce seq_len/pred_len.'
        )
    if flag == 'train' and drop_last and dataset_len < batch_size:
        raise ValueError('Training split is smaller than batch_size; reduce batch_size.')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
