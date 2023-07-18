"""
The work is licensed by Attribution-NonCommercial 4.0 International

Authors:
    Ayhem Bouabid (ayhem18) <bouabidayhem@gmail.com>
    Aleksandr Lobanov (teexone) <dev@alobanov.space>

To get extra information about the project or the license, contact 
any of the following:

    dev@alobanov.space
    license@alobanov.space
"""


"""
Executes service and produces an output video. 

    python -m cntell --help
"""
import argparse


def main():
    argparser = argparse.ArgumentParser(
        description="See and Tell: A tool to describe videos using speech and image recognition.\n\n"
        "To run a pipeline you need to specify the path to the video to describe and the "
        "path to save the output video to. Pipeline uses temporary files to store intermediate "
        "results, you can specify the path to save them to using the --temp argument. You can "
        "also specify the number of cpus to use using the --cpus argument. The --serie argument "
        "specifies the serie to recognize characters from. The --embeddings argument specifies "
        "the folder to search for embeddings for the serie in."
        
    )
    argparser.add_argument("video", help="The path to the video to describe.")
    argparser.add_argument("output", help="The path to save the output video to.")
    argparser.add_argument("--temp", help="The path to save temporary files to.", default=".temp")
    argparser.add_argument("--serie", help="The serie to recognize characters from", type=str, default=None)
    argparser.add_argument("--embeddings", help="The folder to search for embeddings", type=str, default=None)

    args = argparser.parse_args()

    if not hasattr(args, 'h') and not hasattr(args, 'help'):
        from . import run_pipeline
        run_pipeline(
            args.video,
            args.output,
            args.temp,
            args.embeddings,
            args.serie,
        )

    import shutil
    shutil.rmtree(args.temp)

if __name__ == '__main__':
    main()
