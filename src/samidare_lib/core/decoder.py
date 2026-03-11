import click
import pathlib
import sys

this_file_path = pathlib.Path(__file__).parent
sys.path.append(str(this_file_path.parent.parent.parent / "src"))

from samidare_lib.core.decode_v0 import main as decode_v0_main
from samidare_lib.core.decode_v1 import main as decode_v1_main
from samidare_lib.core.pulse_finder import main as pulse_finder_main
from samidare_lib.core.event_builder import main as event_builder_main

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main():
    """Decoder entrypoint."""
    pass

main.add_command(decode_v0_main, name="v0")
main.add_command(decode_v1_main, name="v1")
main.add_command(pulse_finder_main, name="pulse")
main.add_command(event_builder_main, name="event")

if __name__ == "__main__":
    main()