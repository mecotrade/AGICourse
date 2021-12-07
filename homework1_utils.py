import numpy as np

import time
import json
import datetime

from sklearn.manifold import TSNE

import tensorflow as tf

# pip install googletrans==4.0.0rc
# https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
from googletrans import Translator

import matplotlib.pyplot as plt
from IPython.display import clear_output

def create_dict(filename, source_words, target_words, num_words=1000, num_attempts=10, polite_delay=0.25, ban_deplay=10):
    
    translator = googletrans.Translator()
    translations = {}
    progbar = tf.keras.utils.Progbar(num_words)
    for offset, w in enumerate(source_words):
        time.sleep(polite_delay)
        success = False
        for _ in range(num_attempts):
            try:
                translation = translator.translate(w, src='ru', dest='en')
                w_en = translation.text
                success = True
                break
            except Exception as ex:
                time.sleep(ban_deplay)
        assert success, 'After %d attempts translation stil fails' % num_attempts
        if w_en in target_words:
            translations[w] = w_en
            progbar.add(1)
        if len(translations) >= num_words:
            break
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(translations, file, indent=4)
    
    return translations, offset
    
# GloVe vectors http://nlp.stanford.edu/data/glove.6B.zip
def glove_to_gensim_converter(glove_filename, gensim_filename, dimension):
    with open(glove_filename, 'r', encoding='utf-8') as f:
        count = 0
        while True:
            line = f.readline()
            if not line:
                break
            count += 1
    gensim_glove_file = open(gensim_filename, 'w', encoding='utf-8')
    with open(glove_filename, 'r', encoding='utf-8') as f:
        gensim_glove_file.writelines('%d %d\n' % (count, dimension))
        while True:
            line = f.readline()
            if not line:
                break
            gensim_glove_file.writelines(line)
    gensim_glove_file.close()
    
def plot_embeddings(vectors_ru, vectors_en, title, path=None):
    vectors = np.concatenate([vectors_ru, vectors_en])
    vectors_tsne = TSNE(metric='cosine', square_distances=True).fit_transform(vectors)
    plt.figure(figsize=(10, 7))
    plt.title(title, fontsize=24)
    plt.scatter(vectors_tsne[:len(vectors_ru), 0], vectors_tsne[:len(vectors_ru), 1], color='red', label='RU')
    plt.scatter(vectors_tsne[len(vectors_ru):, 0], vectors_tsne[len(vectors_ru):, 1], color='green', label='EN')
    plt.legend(fontsize=16)
    plt.grid(True)
    if path is not None:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
    return vectors_tsne[:len(vectors_ru)], vectors_tsne[len(vectors_ru):]
    
class ProgressIndicator:
    
    def __init__(self, target, step_label='Steps', figsize=(10, 7), margin=0.05, yzero=False, scatter_colors={'RU': 'red', 'EN': 'green'}, scatter_figsize=(10, 5)):
        
        self.target = target
        self.step_label = step_label
        self.figsize = figsize
        self.margin = margin
        self.yzero = yzero
        self.scatter_colors = scatter_colors
        self.scatter_figsize = scatter_figsize
        
        self.steps = []
        self.values = {}
        self.init_ts = datetime.datetime.now()
    
    def plot(self, scatter):
        fig, ax = plt.subplots(len(self.values), 1, sharex=True, figsize=self.figsize)
        if len(self.values) == 1:
            ax = [ax]
        fig.subplots_adjust(hspace=0)
        # plot 
        for n, (label, vals) in enumerate(self.values.items()):
            # set title to first plot
            if n == 0:
                ax[n].set_title(f'Training time: {datetime.datetime.now() - self.init_ts}')
            # data range
            v_min, v_max = (min(vals + [0]), max(vals + [0])) if self.yzero else min(vals)*(1 - 1e-6), max(vals)*(1 + 1e-6)
            v_range = v_max - v_min
            # plot data
            ax[n].set_xlim(0, self.target)
            ax[n].set_ylim(v_min - v_range * self.margin, v_max + v_range * self.margin)
            ax[n].set_xlabel(self.step_label)
            ax[n].set_ylabel(label)
            ax[n].plot(self.steps, vals)
            ax[n].grid(True)
        plt.show()
        
        # plot scatter
        if scatter is not None:
            plt.figure(figsize=self.scatter_figsize)
            for label, points in scatter.items():
                plt.scatter(points[:, 0], points[:, 1], color=self.scatter_colors[label], label=label)
            plt.legend(fontsize=16)
            plt.grid(True)
            plt.show()
        
    def update(self, step, values, scatter):
        # update data accumulator
        self.steps.append(step)        
        for label, val in values.items():
            if label in self.values:
                self.values[label].append(val)
            else:
                self.values[label] = [val]
        # clear existing output
        clear_output(wait=True)
        # plot new graph
        self.plot(scatter)
        
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, max_value = 1e-3, warmup_steps=20000, decay=1):
        
        super(CustomSchedule, self).__init__()

        self.max_value = max_value
        self.warmup_steps = warmup_steps
        self.decay = decay

    def __call__(self, step):
        
        inverse = self.max_value * (1 + self.decay) / (1 + self.decay * step / self.warmup_steps)
        linear = self.max_value * step / self.warmup_steps

        return tf.math.minimum(inverse, linear)