import click

@click.group()
def cli():
    pass

@click.group()
def download():
    """Download data"""
    pass

@click.command()
def all():
    """Download all lookup table data."""
    from .data import DownloadLookupTableData
    DownloadLookupTableData()

@click.command()
def env():
    """Show environment variables and locations"""
    from .data import FWL_DATA_DIR

    click.echo(f'FWL_DATA location: {FWL_DATA_DIR}')

cli.add_command(download)
download.add_command(all)
cli.add_command(env)

if __name__ == '__main__':
    cli()
