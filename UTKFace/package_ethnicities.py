import aisquared
import click

@click.command()
@click.argument('model-file', type = click.Path(exists = True, file_okay = True, dir_okay = False))
@click.option('--dtype', type = str, default = 'float32')
def main(model_file, dtype):
    label_map = [
        'White',
        'Black',
        'Asian',
        'Other'
    ]
    harvester = aisquared.config.harvesting.ImageHarvester()
    preprocesser = aisquared.config.preprocessing.ImagePreprocessor(
        [aisquared.config.preprocessing.Resize([256, 256])]
    )
    analytic = aisquared.config.analytic.LocalModel(model_file, 'cv')
    postprocesser = aisquared.config.postprocessing.MulticlassClassification(
        label_map
    )
    renderer = aisquared.config.rendering.ImageRendering(
        thickness = '10px',
        font_size = '10px'
    )
    feedback = aisquared.config.feedback.MulticlassFeedback(
        label_map
    )
    config = aisquared.config.ModelConfiguration(
        'EthnicityPredicter',
        harvester,
        preprocesser,
        analytic,
        postprocesser,
        renderer,
        feedback
    )
    if dtype != 'float32':
        config.compile(dtype = dtype)
    else:
        config.compile()

if __name__ == '__main__':
    main()
