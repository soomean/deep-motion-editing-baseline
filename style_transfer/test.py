import os
import sys
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
from os.path import join as pjoin
import argparse
import importlib

from data_loader import process_single_bvh, process_single_json

from trainer import Trainer
from remove_fs import remove_fs, save_bvh_from_network_output

source_dir = './TEST_sources/'
reference_dir = './TEST_references/'
stylized_dir = './TEST_stylized/'

def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--content_src', type=str, default=None)
    parser.add_argument('--style_src', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    return parser.parse_args()


def main(args):
    config_module = importlib.import_module(args.config)
    config = config_module.Config()

    # Load experiment setting
    config.initialize(args)

    # Trainer
    trainer = Trainer(config)
    trainer.to(config.device)
    trainer.resume()

    # content_src = 'TEST_sources/fast walking_neutral_01_m_r.bvh'
    # style_src =  'TEST_references/reference-sexy-fast walking.bvh' # "depressed_jumping.bvh"
    # co_data = process_single_bvh(content_src, config, to_batch=True)
    # if style_src.endswith('.bvh'):
    #     status = '3d'
    #     st_data = process_single_bvh(style_src, config, downsample=1, to_batch=True)
    # else:
    #     status = '2d'
    #     st_data = process_single_json(args.style_src, config, to_batch=True)
    # output = trainer.test(co_data, st_data, status)
    # foot_contact = output["foot_contact"][0].cpu().numpy()
    # motion = output["trans"][0].detach().cpu().numpy()
    # output_dir = pjoin(config.main_dir, 'test_output') if args.output_dir is None else args.output_dir
    # save_bvh_from_network_output(motion, output_path=pjoin(output_dir, 'stylized2.bvh'))
    # remove_fs(motion, foot_contact, output_path=pjoin(output_dir, 'stylized2_fs.bvh'))

    sources = get_bvh_files(source_dir)
    references = get_bvh_files(reference_dir)
    for si in range(len(sources)):
        for ri in range(len(references)):
            src = sources[si]
            src_sty = (src.split('/')[-1]).split('_')[1]
            src_con = (src.split('/')[-1]).split('_')[0]
            ref = references[ri]
            ref_sty = (ref.split('/')[-1]).split('-')[1]
            ref_con = (ref.split('/')[-1]).split('-')[2][:-4]

            if src_sty == 'neutral':
                co_data = process_single_bvh(src, config, to_batch=True)
                if ref.endswith('.bvh'):
                    status = '3d'
                    st_data = process_single_bvh(ref, config, downsample=1, to_batch=True)
                else:
                    status = '2d'
                    st_data = process_single_json(args.style_src, config, to_batch=True)
                output = trainer.test(co_data, st_data, status)
                foot_contact = output["foot_contact"][0].cpu().numpy()
                motion = output["trans"][0].detach().cpu().numpy()
                save_bvh_from_network_output(motion, output_path=pjoin(stylized_dir, '%d-stylized-(%s-%s)+(%s-%s).bvh' % (si, src_sty, src_con, ref_sty, ref_con)))
                remove_fs(motion, foot_contact, output_path=pjoin(stylized_dir, '%d-stylized-(%s-%s)+(%s-%s)_fs.bvh' % (si, src_sty, src_con, ref_sty, ref_con)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
