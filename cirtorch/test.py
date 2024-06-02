import argparse
import os
import time
import pickle

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

# Windows
def ensure_path_format(path):
    return path.replace('/', '\\') if os.name == 'nt' else path

# download file using requests
def download_file(url, dest):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f'[requests] File downloaded: {dest}')

def main():
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')
    parser.add_argument('--network-path', type=str, default=None, help='network path, destination where network is saved')
    parser.add_argument('--network-offtheshelf', type=str, default=None, help='network off-the-shelf, in format ARCHITECTURE-POOLING-whiten-RELU')
    parser.add_argument('--datasets', type=str, default='oxford5k,paris6k', help='comma separated list of test datasets')
    parser.add_argument('--image-size', default=1024, type=int, metavar='N', help='maximum size of longer image side used for testing (default: 1024)')
    parser.add_argument('--multiscale', type=str, default='[1]', help='use multiscale vectors for testing, default: single scale (1)')
    parser.add_argument('--whitening', type=str, default=None, help='dataset used to learn whitening for testing')
    parser.add_argument('--gpu-id', type=str, default='0', help='gpu id used for testing (default: 0)')
    args = parser.parse_args()

    datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']

    # Check if there are unknown datasets
    for dataset in args.datasets.split(','):
        if dataset not in datasets_names:
            raise ValueError('Unsupported or unknown dataset: {}!'.format(dataset))

    # check if test dataset are downloaded
    # and download if they are not
    download_train(get_data_root())
    download_test(get_data_root())

    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            state = load_url(PRETRAINED[args.network_path], model_dir=ensure_path_format(os.path.join(get_data_root(), 'networks')))
        else:
            state = torch.load(ensure_path_format(args.network_path))
        net_params = {
            'architecture': state['meta']['architecture'],
            'pooling': state['meta']['pooling'],
            'local_whitening': state['meta'].get('local_whitening', False),
            'regional': state['meta'].get('regional', False),
            'whitening': state['meta'].get('whitening', False),
            'mean': state['meta']['mean'],
            'std': state['meta']['std'],
            'pretrained': False
        }
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        
        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {
            'architecture': offtheshelf[0],
            'pooling': offtheshelf[1],
            'local_whitening': 'lwhiten' in offtheshelf[2:],
            'regional': 'reg' in offtheshelf[2:],
            'whitening': 'whiten' in offtheshelf[2:],
            'pretrained': True
        }
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = list(eval(args.multiscale))
    if len(ms) > 1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp = net.pool.p.item()
        print(">> Set-up multiscale:")
        print(">>>> ms: {}".format(ms))
        print(">>>> msp: {}".format(msp))
    else:
        msp = 1

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # compute whitening
    if args.whitening is not None:
        start = time.time()

        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            Lw = net.meta['Lw'][args.whitening]['ms'] if len(ms) > 1 else net.meta['Lw'][args.whitening]['ss']
        else:
            whiten_fn = ensure_path_format(args.network_path + '_{}_whiten'.format(args.whitening))
            if len(ms) > 1:
                whiten_fn += '_ms'
            whiten_fn += '.pth'
            if whiten_fn and os.path.isfile(whiten_fn):
                    if os.path.isfile(whiten_fn):
                        os.remove(whiten_fn)
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)

            else:
                print('>> {}: Learning whitening...'.format(args.whitening))
                db_root = ensure_path_format(os.path.join(get_data_root(), 'train', args.whitening))
                ims_root = ensure_path_format(os.path.join(db_root, 'ims'))
                db_fn = ensure_path_format(os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening)))
                with open(db_fn, 'rb') as f:
                    db = pickle.load(f)
                images = [ensure_path_format(cid2filename(db['cids'][i], ims_root)) for i in range(len(db['cids']))]
                print('>> {}: Extracting...'.format(args.whitening))
                wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
                print('>> {}: Learning...'.format(args.whitening))
                wvecs = wvecs.numpy()
                m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                Lw = {'m': m, 'P': P}
                if whiten_fn:
                    if os.path.isfile(whiten_fn):
                        os.remove(whiten_fn)
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)
        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets:
        start = time.time()
        print('>> {}: Extracting...'.format(dataset))
        cfg = configdataset(dataset, ensure_path_format(os.path.join(get_data_root(), 'test')))
        images = [ensure_path_format(cfg['im_fname'](cfg, i)) for i in range(cfg['n'])]
        qimages = [ensure_path_format(cfg['qim_fname'](cfg, i)) for i in range(cfg['nq'])]
        try:
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]
        except:
            bbxs = None
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
        print('>> {}: Evaluating...'.format(dataset))

        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

        scores = np.dot(vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)
        # Evaluate ranks
        compute_map_and_print(dataset, ranks, cfg['gnd'])
        if Lw is not None:
            print('>> {}: Evaluating with whitening...'.format(dataset))
            qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])
            vecs_lw = whitenapply(vecs, Lw['m'], Lw['P'])
            scores_lw = np.dot(vecs_lw.T, qvecs_lw)
            ranks_lw = np.argsort(-scores_lw, axis=0)
            compute_map_and_print(dataset + ' + whiten', ranks_lw, cfg['gnd'])
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time() - start)))

def download_train(data_root):
    train_dir = ensure_path_format(os.path.join(data_root, 'train', 'retrieval-SfM-120k'))
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    ims_dir = ensure_path_format(os.path.join(train_dir, 'ims'))
    if not os.path.isdir(ims_dir):
        os.makedirs(ims_dir)
        print(f">> Image directory does not exist. Creating: {ims_dir}")
        ims_url = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/train/retrieval-SfM-120k/ims.tar.gz"
        ims_tar_path = ensure_path_format(os.path.join(train_dir, 'ims.tar.gz'))
        print(f">> Downloading {ims_url}...")
        download_file(ims_url, ims_tar_path)
        # Extracting tar.gz (requires tarfile module)
        import tarfile
        with tarfile.open(ims_tar_path, "r:gz") as tar:
            tar.extractall(path=train_dir)
        os.remove(ims_tar_path)

def download_test(data_root):
    test_dir = ensure_path_format(os.path.join(data_root, 'test'))
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
    test_datasets = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
    for dataset in test_datasets:
        dataset_dir = ensure_path_format(os.path.join(test_dir, dataset))
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
            print(f">> Image directory does not exist. Creating: {dataset_dir}")
            ims_url = f"http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/{dataset}/ims.tar.gz"
            ims_tar_path = ensure_path_format(os.path.join(dataset_dir, 'ims.tar.gz'))
            print(f">> Downloading {ims_url}...")
            download_file(ims_url, ims_tar_path)
            # Extracting tar.gz (requires tarfile module)
            import tarfile
            with tarfile.open(ims_tar_path, "r:gz") as tar:
                tar.extractall(path=dataset_dir)
            os.remove(ims_tar_path)

if __name__ == '__main__':
    main()
