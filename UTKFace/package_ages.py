import aisquared
import click

@click.command()
@click.argument('model-file', type = click.Path(exists = True, file_okay = True, dir_okay = False))
@click.option('--dtype', type = str, default = 'float32')
def main(model_file, dtype):
    harvester = aisquared.config.harvesting.ImageHarvester()
    preprocesser = aisquared.config.preprocessing.image.ImagePreprocessor(
        [
            aisquared.config.preprocessing.image.Resize([256, 256]),
            aisquared.config.preprocessing.image.DivideValue(255)
        ]
    )
    analytic = aisquared.config.analytic.LocalModel(model_file, 'cv')
    postprocesser = aisquared.config.postprocessing.Regression()
    renderer = aisquared.config.rendering.ImageRendering(
        thickness = '10',
        font_size = '10'
    )
    feedback = aisquared.config.feedback.RegressionFeedback()
    config = aisquared.config.ModelConfiguration(
        'AgePredicter',
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
