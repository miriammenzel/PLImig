#!/usr/bin/env python3
import argparse
import numpy
import os
import plimg
import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description='Extracting parameters from '
                                                 'transmittance and retardation'
                                                 ' images for creation of '
                                                 'inclination images.')

    parser.add_argument('-i', '--input', nargs='*',
                        help='Input transmittance files.', required=True)
    parser.add_argument('-o', '--output',
                        help='Output folder', required=True)
    parser.add_argument('--detailed',
                        action='store_true')
    parser.add_argument('--with_blurred',
                        action='store_true')
    parser.add_argument('--dataset',
                        type=str,
                        required=False,
                        default="/Image")

    arguments = parser.parse_args()
    args = vars(arguments)

    reader = plimg.reader.FileReader()
    writer = plimg.writer.HDF5Writer()

    paths = args['input']
    if not isinstance(paths, list):
        paths = [paths]

    if not os.path.exists(args['output']):
        os.makedirs(args['output'], exist_ok=True)

    tqdm_paths = tqdm.tqdm(paths, leave=True)
    tqdm_step = tqdm.tqdm(total=9, leave=True)
    for path in tqdm_paths:
        transmittance_path = path
        retardation_path = transmittance_path.replace('median10', '')\
            .replace('NTransmittance', 'Transmittance')\
            .replace('Transmittance', 'Retardation')

        # Get slice name of output_folder
        slice_name = transmittance_path[transmittance_path.rfind('/') + 1:-3]
        slice_name = slice_name.replace('median10', '')\
            .replace('NTransmittance', 'Transmittance')\
            .replace('Transmittance_', '')
        tqdm_paths.set_description(slice_name)

        tqdm_step.set_description('Read transmittance')
        reader.set(transmittance_path, dataset=args['dataset'])
        transmittance = reader.get()
        tqdm_step.update()

        tqdm_step.set_description('Read retardation')
        reader.set(retardation_path, dataset=args['dataset'])
        retardation = reader.get()
        tqdm_step.update()

        tqdm_step.set_description('Generating median10Transmittance')
        if 'median10' not in transmittance_path:
            med_transmittance = plimg.filters.median(transmittance, 10)
        else:
            med_transmittance = transmittance
        tqdm_step.update()

        generation = plimg.mask.\
            MaskGeneration(med_transmittance, retardation)

        writer.set_path(args['output'] + '/' + slice_name + '.h5')
        tqdm_step.set_description('Setting attributes')
        writer.set_attributes('/', generation.t_tra, generation.t_ret,
                               generation.t_min, generation.t_max)
        params = numpy.array([generation.t_tra, generation.t_ret,
                               generation.t_min, generation.t_max]) \
                      .reshape((4, 1))
        writer.set_dataset(dataset=args['dataset']+"/Params",
                           content=params,
                           dtype=numpy.float32)
        tqdm_step.update()

        tqdm_step.set_description('Generating and writing white mask')
        writer.set_dataset(dataset=args['dataset']+"/White",
                           content=generation.white_mask*255,
                           dtype=numpy.uint8)
        tqdm_step.update()

        tqdm_step.set_description('Generating and writing gray mask')
        writer.set_dataset(dataset=args['dataset']+"/Gray",
                           content=generation.gray_mask*255,
                           dtype=numpy.uint8)
        tqdm_step.update()

        if args['with_blurred']:
            tqdm_step.set_description('Generating and writing difference mask')
            writer.set_dataset(dataset=args['dataset']+"/Blurred",
                               content=generation.blurred_mask,
                               dtype=numpy.float32)
        tqdm_step.update()

        if args['detailed']:
            tqdm_step.set_description('Generating and writing gray mask')
            writer.set_dataset(dataset=args['dataset']+"/No_Nerve_Fibers",
                               content=generation.no_nerve_fiber_mask*255,
                               dtype=numpy.uint8)

            if not numpy.all(med_transmittance == transmittance):
                tqdm_step.set_description('Writing median transmittance')
                writer.set_dataset(dataset=args['dataset']+"/med10Transmittance",
                                   content=med_transmittance,
                                   dtype=numpy.float32)
        tqdm_step.update()
        tqdm_step.reset()


if __name__ == "__main__":
    main()
