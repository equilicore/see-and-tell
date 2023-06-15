from src import run_pipeline
import argparse

if __name__ == '__main__':
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
    argparser.add_argument("--cpus", help="The number of cpus to use.", type=int, default=1)
    argparser.add_argument("--serie", help="The serie to recognize characters from", type=str, default=None)
    argparser.add_argument("--embeddings", help="The folder to search for embeddings", type=str, default=None)

    args = argparser.parse_args()

    run_pipeline(
        args.video,
        args.output,
        args.temp,
        args.cpus,
        args.embeddings,
        args.serie,
    )

    import shutil
    shutil.rmtree(args.temp)
