import ffmpeg
from argparse import ArgumentParser

def trim_video(in_file, out_file, start=0, end=7500):
    input_stream = ffmpeg.input(in_file)
    trimmed_stream = input_stream.trim(start_frame=start, end_frame=end)
    # Output the trimmed video
    output_stream = ffmpeg.output(trimmed_stream, out_file)
    # Run the FFmpeg command
    ffmpeg.run(output_stream)


def main():
    parser = ArgumentParser()
    parser.add_argument('--inp', type=str, help='Input video path')
    parser.add_argument('--out', type=str, help='Output, trimmed video path')
    parser.add_argument('--start', type=int, default=0, help='Start frame')
    parser.add_argument('--end', type=int, default=7500,help='End frame')
    args = parser.parse_args()

    inpath = args.inp
    outpath = args.out
    start_f = args.start
    end_f = args.end
     
    trim_video(inpath, outpath, start_f, end_f)


if __name__ == '__main__':
    main()
