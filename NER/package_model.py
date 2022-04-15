import aisquared
import json

with open('vocab.json', 'r') as f:
    vocab = json.load(f)

harvester = aisquared.config.harvesting.TextHarvester()
preprocesser = aisquared.config.preprocessing.TextPreprocessor(
    [
        aisquared.config.preprocessing.RemoveCharacters(),
        aisquared.config.preprocessing.ConvertToCase(),
        aisquared.config.preprocessing.Tokenize(split_sentences = True),
        aisquared.config.preprocessing.ConvertToVocabulary(
            vocab,
            0,
            1
        ),
        aisquared.config.preprocessing.PadSequences(
            0,
            32,
            'pre',
            'post'
        )
    ]
)
model = aisquared.config.analytic.LocalModel('NERModel.h5', 'text')
postprocesser = aisquared.config.postprocessing.BinaryClassification(
    [
        'Not Entity',
        'Entity'
    ]
)
renderer = aisquared.config.rendering.WordRendering(result_key = 'className')
model_feedback = aisquared.config.feedback.ModelFeedback()
model_feedback.add_question('Is this model accurate?', choices = ['yes', 'no'])

config = aisquared.config.ModelConfiguration(
    'NERModel',
    harvester,
    preprocesser,
    model,
    postprocesser,
    renderer,
    model_feedback
).compile(dtype = 'float16')