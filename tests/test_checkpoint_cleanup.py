from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from src import train_config, train_profile


@pytest.mark.parametrize('exit_code,options,removed', [
    (0, [], True),
    (1, [], False),
    (0, ['--cleanup-checkpoints', 'false'], False),
])
def test_cleanup_after_training(tmp_path, monkeypatch, exit_code, options, removed):
    output = tmp_path / 'run'
    checkpoint = output / 'checkpoint-500'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.safetensors').write_text('snapshot')
    (checkpoint / 'optimizer.pt').write_text('optimizer')
    final = output / 'model.safetensors'
    final.write_text('final')
    unrelated = output / 'checkpoint-notes'
    unrelated.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'keep').write_text('keep')
    (output / 'checkpoint-1000').symlink_to(outside, target_is_directory=True)
    config = tmp_path / 'train.yaml'
    config.write_text(f'output_dir: {output}\n')
    monkeypatch.setattr(train_config, 'require_training_inputs', lambda _: None)
    monkeypatch.setattr(train_config, 'prepare_environment', lambda *_: {})
    monkeypatch.setattr(train_config.subprocess, 'run',
                        lambda *a, **kw: SimpleNamespace(returncode=exit_code))
    result = CliRunner().invoke(train_config.main, [str(config), *options])
    assert result.exit_code == exit_code, result.output
    assert checkpoint.exists() is not removed
    assert final.read_text() == 'final'
    assert unrelated.is_dir()
    assert (outside / 'keep').read_text() == 'keep'
    assert (output / 'checkpoint-1000').is_symlink()
    if exit_code:
        assert (checkpoint / 'optimizer.pt').exists()


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_profile_forwards_checkpoint_option(monkeypatch, enabled):
    monkeypatch.setattr(train_profile, 'generated_configs', lambda *_: ['train.yaml'])
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_profile.subprocess, 'run', run)
    result = CliRunner().invoke(train_profile.main, ['--cleanup-checkpoints', enabled])
    assert result.exit_code == 0, result.output
    command = commands[0]
    assert command[command.index('--cleanup-checkpoints') + 1].lower() == enabled
