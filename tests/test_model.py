import sys
import tempfile
import zipfile
import os
import json
import io

if sys.version_info.minor > 6:
    python_36 = False
    from contextlib import nullcontext
else:
    python_36 = True
    from contextlib import suppress as nullcontext
from xrnn.data_handler import DataHandler
from xrnn import layer_utils
from xrnn.model import Model
from xrnn import activations
from xrnn import optimizers
from xrnn import losses
from xrnn import config
from xrnn import layers
from xrnn import ops
import re
import pytest


BATCH_SIZE = 64
INPUT_SHAPE = (BATCH_SIZE, 28, 28, 3)
IMAGE_BATCH = ops.zeros(INPUT_SHAPE)
LABELS = ops.ones((BATCH_SIZE,))
X = ops.random.standard_normal((200, *INPUT_SHAPE[1:]))
Y = ops.ones(len(X))


def one_hot_encoded(n_classes):
    integers = ops.random.integers(0, n_classes, (BATCH_SIZE,))
    one_hot = ops.zeros((BATCH_SIZE, n_classes))
    one_hot[ops.arange(len(integers)), integers] = 1
    return one_hot


@pytest.fixture
def cnn():
    m = Model()
    m.add(layers.Conv2D(16, 3, 2))
    m.add(layers.BatchNormalization())
    m.add(activations.LeakyReLU())
    m.add(layers.Dropout(0.25))
    m.add(layers.AvgPooling2D(2, 2, 'same'))
    m.add(layers.Flatten())
    m.add(layers.Dense(10))
    m.add(activations.Softmax())
    m.set('adam', 'categorical_crossentropy')
    return m


class TestModel:

    @pytest.mark.parametrize(
        "opt, loss, raises",
        [
            ('adam', 'mse', nullcontext()),
            ('Adam', 'MSE', nullcontext()),
            ('rmsprop', 'rmsprop', pytest.raises(ValueError)),
            ('adam', 'categorical_crossentropy', nullcontext()),
            (optimizers.Adam(), 'mse', nullcontext()),
            ('sgd', losses.BinaryCrossentropy(), nullcontext()),
            ('adagrad', 'binary_crossentropy', nullcontext()),
            ('adamm', losses.CategoricalCrossentropy(), pytest.raises(ValueError))
        ]
    )
    def test_set(self, opt, loss, raises):
        with raises:
            model = Model()
            model.set(opt, loss)
            if isinstance(opt, str):
                opt_str = {
                    'adam': optimizers.Adam, 'sgd': optimizers.SGD,
                    'rmsprop': optimizers.RMSprop, 'adagrad': optimizers.Adagrad}
                opt = opt_str[opt.lower()]()
            if isinstance(loss, str):
                loss_str = {
                    'mse': losses.MSE,
                    'categorical_crossentropy': losses.CategoricalCrossentropy,
                    'binary_crossentropy': losses.BinaryCrossentropy}
                loss = loss_str[loss.lower()]()
            assert isinstance(model.optimizer, type(opt))
            assert isinstance(model.loss, type(loss))
            assert model.loss.trainable_layers == model.trainable_layers
            # Test set after add
            model = Model()
            model.add(layers.Dense(10))
            model.set('adam', 'mse')
            assert model.loss.trainable_layers == model.trainable_layers

    def test_add(self):
        m = Model()
        m.set('adam', 'mse')
        m.add(layers.Dense(10))
        m.add(layers.Flatten())
        assert len(m.layers) == 2
        assert len(m.trainable_layers) == 1
        assert len(m.loss.trainable_layers) == 1
        # Test add after build
        m.build((BATCH_SIZE, 100))
        m.add(layers.Dense(16))
        assert m.layers[-1].built
        assert m.layers[-1].weights.shape == (10, 16)
        # Test adding a built layer to a built model with wrong shape.
        d = layers.Dense(16)
        d.build((BATCH_SIZE, 100))
        with pytest.raises(ValueError):
            m.add(d)
        # Test adding a built layer to a built model with correct shape.
        d = layers.Dense(32)
        d.build((BATCH_SIZE, 16))
        m.add(d)
        assert m.output_shape == (BATCH_SIZE, 32)

    def test_build(self):
        m = Model()
        m.add(layers.Dense(16))
        m.add(activations.ReLU())
        m.add(layers.Dense(128))
        m.build((BATCH_SIZE, 100))
        assert m.layers[0].built
        assert m.layers[0].weights.shape == (100, 16)
        assert m.layers[0].weights.max() <= ops.sqrt(6 / 100)  # Make sure 'he' initialization method is used
        assert m.layers[-1].built
        assert m.layers[-1].weights.shape == (16, 128)
        assert m.layers[-1].weights.max() <= ops.sqrt(6 / (16 + 128))  # Make sure 'xavier' method is used.

        # Test adding an already built layer then calling model.build().
        m = Model()
        layer = layers.Dense(16)
        layer.build((BATCH_SIZE, 100))
        m.add(layer)
        # Test wrong input shape
        with pytest.raises(ValueError):
            m.build((BATCH_SIZE, 128))
        # Test correct input shape
        m.build((BATCH_SIZE, 100))
        assert m.output_shape == (BATCH_SIZE, 16)

        # Test wrong input shape for layers that don't have a build method but except input_shape to be in a certain way
        m = Model()
        m.add(layers.MaxPooling2D(3))
        with pytest.raises(ValueError):
            m.build((BATCH_SIZE, 100))  # inputs must be 4D
        m.build(INPUT_SHAPE)
        assert m.layers[0].input_shape is not None

    def test_output_shape(self):
        config.set_image_data_format('channels-last')
        m = Model()
        with pytest.raises(ValueError):
            _ = m.output_shape
        m.add(layers.Conv2D(32, 3, 2))
        m.add(layers.MaxPooling2D(2, 2, 'same'))
        m.build((BATCH_SIZE, 64, 64, 3))
        assert m.layers[-1].input_shape == m.layers[0].output_shape == (BATCH_SIZE, 31, 31, 32)
        assert m.output_shape == m.layers[-1].output_shape == (BATCH_SIZE, 16, 16, 32)
        m.add(layers.Flatten())
        assert m.layers[-1].input_shape == (BATCH_SIZE, 16, 16, 32)
        assert m.output_shape == m.layers[-1].output_shape == (BATCH_SIZE, 16 * 16 * 32)

    def test_forward(self, cnn):
        cnn.build(INPUT_SHAPE)
        with pytest.raises(ValueError):
            cnn.forward(ops.zeros((BATCH_SIZE, 7, 7, 3)), True)
        output = cnn.forward(IMAGE_BATCH, False)
        assert output.shape == cnn.output_shape
        assert output.shape == (BATCH_SIZE, 10)
        assert output.dtype == 'float32'
        # 10 classes, softmax activation function, each class has a 0.1 probability when the model isn't trained.
        assert output.max() == pytest.approx(0.1)
        assert cnn.layers[0].inputs is None
        assert cnn.layers[1].training
        cnn.forward(IMAGE_BATCH, True)
        assert cnn.layers[0].inputs is IMAGE_BATCH
        config.set_default_dtype('float64')
        output = cnn.forward(IMAGE_BATCH[:32], False)
        assert cnn.layers[0].inputs is not IMAGE_BATCH  # The datatype changed, so it shouldn't be the same array.
        assert output.dtype == config.DTYPE

    def test_call(self, cnn):
        output = cnn(IMAGE_BATCH)
        assert ops.array_equal(output, cnn.forward(IMAGE_BATCH))

    @pytest.mark.parametrize('labels', [LABELS, one_hot_encoded(10)])
    def test_backward_fast_path(self, labels, cnn):
        # backward has a fast path when the last layer is Softmax and loss is categorical crossentropy.
        y_pred = cnn.forward(IMAGE_BATCH, False).astype('f4')
        with pytest.raises(AttributeError):
            cnn.backward(labels, y_pred)
        cnn.forward(IMAGE_BATCH, True)
        cnn.layers[0].training = False
        config.set_default_dtype('f4')
        output = cnn.backward(labels, y_pred)
        assert output.shape == cnn.input_shape
        assert output.dtype == 'float32'

    @pytest.mark.parametrize('loss', ['mse', 'binary_crossentropy', 'categorical_crossentropy'])
    def test_backward_slow_path(self, loss, cnn):
        # Remove the last (Softmax) layer so the slower path is used when the loss is categorical crossentropy,
        # otherwise, it wouldn't matter.
        if loss == 'categorical_crossentropy':
            cnn.layers.pop()
        cnn.set(cnn.optimizer, loss)
        y_pred = cnn.forward(IMAGE_BATCH)
        assert cnn.backward(y_pred.copy(), y_pred).shape == cnn.input_shape

    @pytest.mark.parametrize('batch_size', [16, 32, BATCH_SIZE, 128])
    def test_update_progressbar(self, batch_size, cnn, capfd):
        cnn.train(X, Y, batch_size, 1)
        out, err = capfd.readouterr()
        # From 1 to -1, because the 0th element is the epoch counter, and the last element is the same progressbar with
        # different formatting.
        updates = out.split('\r')[1:-1]
        for i, update in enumerate(updates):
            assert update.split()[1] == f'{i + 1}/{len(updates)}'  # number of steps
            assert len(update.split()[2]) == 32  # The progressbar part
            assert update.split('-')[1].strip() == f'{((i + 1) / len(updates)):.0%}'
        assert 'Took' in out.split('\r')[-1]
        # Test that only two new lines were printed, one after the epoch counter and one after the progressbar.
        assert out.count('\n') == 2
        cnn.train(X, Y, batch_size, 2, validation_split=0.1)
        out, err = capfd.readouterr()
        assert out.count('val_loss') == 2
        assert out.count('Took') == 2
        # 6 newlines, 2 for the first epoch and progressbar, 1 line between them and the second epoch, 2 for the 2nd
        # epoch, 1 for print(Training took).
        assert out.count('\n') == 6

    @pytest.mark.parametrize(
        "val_split, val_data",
        [
            (0.1, None),
            (0.1, (IMAGE_BATCH, ops.ones(len(IMAGE_BATCH)))),
            (0., [IMAGE_BATCH, ops.ones(len(IMAGE_BATCH))]),
            (None, (IMAGE_BATCH, ops.ones(len(IMAGE_BATCH)))),
            (None, DataHandler(IMAGE_BATCH, ops.ones(len(IMAGE_BATCH)))),
            (None, None),
            (0., None)
        ]
    )
    def test_train_validation_data(self, val_split, val_data, cnn, capfd):
        cnn.train(X, Y, BATCH_SIZE, validation_split=val_split, validation_data=val_data)
        out, err = capfd.readouterr()
        if any([val_split, val_data]):
            assert 'val_loss' in out
        else:
            assert 'val_loss' not in out
        # Test validation_split when x is a DataHandler object and not an array.
        if val_split and not val_data:
            with pytest.raises(TypeError):
                cnn.train(DataHandler(X, Y), validation_split=val_split, validation_data=val_data)

    @pytest.mark.parametrize(
        "epochs, print_every, valid_freq, should_print",
        [
            (1, 1, 1, True),
            (1, 2, 0, False),
            (2, 0, 0, False),
            (5, False, False, False),
            (1, None, None, False),
            (2, 2, 0, True),
            (2, 1, 3, True),
            (6, 2, 3, True),
        ]
    )
    def test_train_verbose(self, epochs, print_every, valid_freq, should_print, cnn, capfd):
        cnn.train(X, Y, BATCH_SIZE, epochs, print_every=print_every, validation_split=0.1, validation_freq=valid_freq)
        out, err = capfd.readouterr()
        assert bool(out) == should_print
        if should_print:
            # Test that is printed every 'print_every'th epoch.
            assert out.count('Epoch') == epochs // print_every
            assert bool('Training took' in out) == bool(epochs >= 2)  # It should print that only if epochs>=2
            if valid_freq and valid_freq * print_every <= epochs:
                assert 'val_loss' in out
            else:
                assert 'val_loss' not in out

    @pytest.mark.parametrize("steps_per_epoch", [1, 2, 3, 4, 5])
    def test_train_steps_per_epoch(self, steps_per_epoch, cnn, capfd):
        cnn.train(X, Y, BATCH_SIZE, steps_per_epoch=steps_per_epoch)
        out, err = capfd.readouterr()
        assert cnn.optimizer.iterations == min(steps_per_epoch, len(X) // BATCH_SIZE + 1)
        # Test that the progressbar print the right number of steps.
        assert out.splitlines()[2].split('/')[1][0] == str(min(steps_per_epoch, len(X) // BATCH_SIZE + 1))

    def test_train_optimizer_updates(self, cnn):
        m = Model()
        # Test calling train before `set`.
        with pytest.raises(ValueError):
            m.train(X, Y)
        cnn.layers[0].training = False
        cnn.build(INPUT_SHAPE)
        before_weights = [cnn.layers[0].weights.copy(), cnn.layers[1].weights.copy()]
        cnn.train(X, Y, BATCH_SIZE)
        # Test that the optimizer didn't update non-trainable layers.
        assert ops.array_equal(cnn.layers[0].weights, before_weights[0])
        # Test that the optimizer updates trainable layers.
        assert not ops.array_equal(cnn.layers[1].weights, before_weights[1])

    def test_evaluate(self, cnn):
        loss, acc = cnn.evaluate(IMAGE_BATCH, LABELS).values()
        assert cnn.layers[0].inputs is None
        assert loss > 2
        assert acc == 0

    def test_inference(self, cnn):
        output = cnn.inference(IMAGE_BATCH)
        assert output.shape == (INPUT_SHAPE[0], *cnn.output_shape[1:])
        output = cnn.inference(IMAGE_BATCH, 6)
        assert output.shape == (INPUT_SHAPE[0], *cnn.output_shape[1:])

    def test_mem_usage(self, cnn):
        with pytest.raises(ValueError):
            cnn.mem_usage()
        cnn.build(INPUT_SHAPE)
        config.set_default_dtype('f4')
        mem_usage = cnn.mem_usage()
        assert mem_usage == cnn.mem_usage(BATCH_SIZE)
        assert mem_usage < cnn.mem_usage(BATCH_SIZE * 2)
        config.set_default_dtype(float)
        assert cnn.mem_usage() == mem_usage * 2
        config.set_default_dtype('f4')
        # Test optimizers other than adam mem usage.
        cnn.set('adagrad', 'categorical_crossentropy')
        assert cnn.mem_usage() < mem_usage

    def test_summary(self, capfd, cnn):
        # Test calling `summary()` with a big batch size that isn't multiple of 2.
        m = Model()
        m.add(layers.MaxPooling2D(2))
        m.build((1025, 28, 28, 3))
        with pytest.warns(UserWarning):
            m.summary()
        # call `readouterr()` to clear it for the next print
        out, err = capfd.readouterr()
        # Test calling `summary()` before `build()`.
        with pytest.raises(ValueError):
            cnn.summary()
        cnn.build(INPUT_SHAPE)
        cnn.summary()
        out, err = capfd.readouterr()
        lines = [line for line in out.splitlines() if line]
        # Got the re pattern from ChatGPT; first time I used something that came out of it, the memes are actually True.
        first_layer_line = re.match(r'(\S+)\s+(\d+)\s+(\([\d\s,]+\))\s+(\([\d\s,]+\))', lines[3])
        assert first_layer_line[1] == cnn.layers[0].name
        assert first_layer_line[2] == str(cnn.layers[0].weights.size + cnn.layers[0].biases.size)
        assert first_layer_line[3] == str(cnn.input_shape)
        assert first_layer_line[4] == str(cnn.layers[0].output_shape)
        assert '-' in lines[0]
        assert '-' in lines[-1]
        assert '=' in lines[2]
        assert '=' in lines[-8]
        assert len(lines[0]) == len(lines[-1]) == len(lines[2]) == len(lines[-8])
        assert len([line for line in out.split('=' * len(lines[2]))[1].splitlines() if line]) == len(cnn.layers)
        assert lines[-2].split(':')[-1].strip() == layer_utils.to_readable_unit_converter(cnn.mem_usage())
        non_trainable_params_before = lines[-6].split(':')[-1].strip()
        trainable_params_before = lines[-7].split(':')[-1].strip()
        cnn.layers[0].training = False
        cnn.summary()
        out, err = capfd.readouterr()
        lines = [line for line in out.splitlines() if line]
        # Test that setting a layer training state changes the parameter count correctly
        assert lines[-6].split(':')[-1].strip() > non_trainable_params_before
        assert lines[-7].split(':')[-1].strip() < trainable_params_before

    def test_get_params(self, cnn):
        m = Model()
        # Test get params on an empty model.
        with pytest.raises(ValueError):
            m.get_params()
        # Test get params on a model with no trainable layers.
        m.add(layers.MaxPooling2D((2, 2)))
        with pytest.raises(ValueError):
            m.get_params()
        # Test get params before build.
        with pytest.raises(ValueError):
            cnn.get_params()
        cnn.build(INPUT_SHAPE)
        model_parameters = cnn.get_params()
        assert len(model_parameters) == 3

    def test_set_params(self, cnn):
        with pytest.raises(ValueError):
            cnn.set_params([[] for _ in range(len(cnn.trainable_layers))])
        cnn.build(INPUT_SHAPE)
        model_parameters = cnn.get_params()
        # Test that setting params actually works and changes the model's layers' parameters.
        cnn.layers[0].set_params([ops.ones_like(cnn.layers[0].weights)])
        cnn.set_params(model_parameters)
        assert ops.array_equal(cnn.layers[0].weights, model_parameters[0][0])
        with pytest.raises(IndexError):
            cnn.set_params(model_parameters[:-1])
        with pytest.raises(IndexError):
            model_parameters.append([])
            cnn.set_params(model_parameters)

    def test_save_params(self, cnn):
        cnn.build(INPUT_SHAPE)
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_params.npz')
        cnn.save_params(temp_file_path)
        try:
            with ops.load(temp_file_path) as model_params_dict:
                assert len(model_params_dict) == 8
                assert len([key for key in model_params_dict if "Conv2D" in key]) == 2
                assert len([key for key in model_params_dict if "BatchNormalization" in key]) == 4
                assert len([key for key in model_params_dict if "Dense" in key]) == 2
                assert len([key for key in model_params_dict if key.endswith('0')]) == 3
                assert all([model_params_dict[key].dtype == config.DTYPE] for key in model_params_dict)
        finally:  # To ensure the file gets deleted.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_load_params(self, cnn):
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_params.npz')
        try:
            cnn.build(INPUT_SHAPE)
            cnn.save_params(temp_file_path)
            cnn.load_params(temp_file_path)
            # Test less params saved than layers.
            cnn.add(layers.Dense(16))
            with pytest.raises(IndexError):
                cnn.load_params(temp_file_path)
            # Test more params saved than layers.
            cnn.layers.pop()
            cnn.layers.pop()
            with pytest.raises(IndexError):
                cnn.load_params(temp_file_path)
        finally:  # To ensure the file gets deleted.
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_save(self, cnn):
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_model.xmodel')
        try:
            with pytest.raises(ValueError):
                cnn.save(temp_file_path)
            cnn.build(INPUT_SHAPE)
            cnn.save(temp_file_path)
            assert os.path.exists(temp_file_path)
            with zipfile.ZipFile(temp_file_path) as saved_model:
                assert saved_model.namelist() == ['model_config.json', 'model_params.npz']
                with saved_model.open(
                        'model_config.json') as config_file, saved_model.open('model_params.npz') as model_params_file:
                    model_config = json.load(config_file)
                    # The following is to convert all lists in dict to tuples for all dicts in a list.
                    # This is a single line, but I split to not violate PEP8 (probably the least of your concerns
                    # when looking at it).
                    # Writing a loop that accomplishes the same is easier and more readable, but I like this.
                    model_config['layers_configs'] = [
                        dict(
                            zip(
                                a.keys(),
                                list(
                                    map(
                                        lambda x: tuple(x) if isinstance(x, list) and len(x) > 1 else x,
                                        a.values()
                                    )
                                )
                            )
                        ) for a in model_config['layers_configs']
                    ]
                    # Test that the model's config was saved correctly.
                    assert model_config == cnn.get_config()
                    if python_36:
                        io.BytesIO(model_params_file.read())
                    loaded_params = ops.load(model_params_file)
                    assert len(loaded_params) == 8  # 2 for conv, 2 for dense, 4 for batch norm.
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_load(self, cnn):
        # `Model.load()` depends only on `zipfile.ZipFile()`, `json.load` (both hopefully tested thoroughly),
        # `Model.from_config()`, `Model.load_params()` (tested separately), so not much to test here apart from testing
        # that the pieces work together.
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_model.xmodel')
        try:
            cnn.build(INPUT_SHAPE)
            cnn.save(temp_file_path)
            new_cnn = cnn.load(temp_file_path)
            assert len(new_cnn.layers) == len(cnn.layers)
            assert new_cnn.built
            assert new_cnn.output_shape == cnn.output_shape
            # Tests that the layers are constructed with the same arguments as the orginal one.
            assert new_cnn.get_config() == cnn.get_config()
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_get_config(self):
        model = Model()
        model_config = model.get_config()
        assert list(model_config.keys()) == ['layers_configs', 'opt_config', 'loss']
        assert len(model_config['layers_configs']) == 0
        assert model_config['opt_config'] is None
        assert model_config['loss'] is None
        model.add(layers.Dense(16))
        model.set('adam', 'mse')
        model_config = model.get_config()
        assert model_config['layers_configs'][0] == model.layers[0].get_config()
        assert model_config['opt_config'] == optimizers.Adam().get_config()
        assert model_config['loss'] == 'MSE'

    @pytest.mark.parametrize('build', [False, True])
    def test_from_config(self, build, cnn):
        if build:
            cnn.build(INPUT_SHAPE)
        model_config = cnn.get_config()
        new_cnn = Model.from_config(model_config, build)
        assert len(new_cnn.layers) == len(cnn.layers)
        assert new_cnn.optimizer.get_config() == cnn.optimizer.get_config()
        assert type(new_cnn.loss) is type(cnn.loss)
        assert new_cnn.built is build
        for new_layer, old_layer in zip(new_cnn.layers, cnn.layers):
            assert new_layer.get_config() == old_layer.get_config()
        if build:
            assert new_cnn.output_shape == cnn.output_shape
            new_cnn.forward(IMAGE_BATCH)
        # Test when the optimizer and loss aren't set.
        model_config['opt_config'] = None
        new_cnn = Model.from_config(model_config)
        assert new_cnn.optimizer is new_cnn.loss is None
        # Test from_config with a custom layer.
        model_config['layers_configs'].append({"type": "MyCustomLayer"})
        with pytest.raises(NotImplementedError):
            Model.from_config(model_config)
