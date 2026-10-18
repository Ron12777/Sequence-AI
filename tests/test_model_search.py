from collections import Counter
import torch
from src.game import SequenceGame
from src.model import SequenceNet, load_model, save_model
from src.mcts import MCTS, MCTSNode


def test_checkpoint_roundtrip_and_compiled_prefix(tmp_path):
    model = SequenceNet(num_filters=8, num_res_blocks=1).eval()
    path = tmp_path / "model.pt"
    save_model(model, path, epoch=3)
    loaded, device, metadata = load_model(path, device="auto", compile_model=False)
    assert metadata["epoch"] == 3
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(
            loaded.state_dict()[name].cpu(), parameter, rtol=0, atol=0
        )
    # Compare execution on the same device; CUDA TF32 and CPU FP32 kernels differ.
    model.to(device)
    x = torch.randn(2, 11, 10, 10, device=device)
    with torch.no_grad():
        expected = model(x)
        actual = loaded.eval()(x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["model_state_dict"] = {
        "_orig_mod." + k: v for k, v in checkpoint["model_state_dict"].items()
    }
    torch.save(checkpoint, path)
    load_model(path, device="cpu", compile_model=False)


def test_channel_migration_preserves_predictions(tmp_path):
    model = SequenceNet(input_channels=8, num_filters=8, num_res_blocks=1).eval()
    path = tmp_path / "old.pt"
    save_model(model, path)
    loaded, _, _ = load_model(path, device="cpu", compile_model=False)
    x = torch.randn(2, 11, 10, 10)
    with torch.no_grad():
        for a, b in zip(model(x[:, :8]), loaded.eval()(x)):
            torch.testing.assert_close(a, b)


def test_determinization_preserves_hidden_card_multiset():
    game = SequenceGame()
    game.reset()
    search = MCTS(SequenceNet(num_filters=8, num_res_blocks=1), torch.device("cpu"))
    result = search._determinize(game, 1)
    assert result.hands[1] == game.hands[1]
    assert Counter(result.hands[2] + result.deck.cards) == Counter(
        game.hands[2] + game.deck.cards
    )


def test_python_values_are_from_the_player_to_move():
    game = SequenceGame()
    game.reset()
    root = MCTSNode(game)
    child_game = game.clone()
    child_game.current_player = 2
    child = MCTSNode(child_game, parent=root, prior=0)
    search = MCTS.__new__(MCTS)
    search._backpropagate([root, child], -1, 2)
    assert root.q_value == 1
    assert child.q_value == -1
    assert child.ucb_score == 1


def test_python_search_prefers_immediate_win():
    from src.cards import Card

    class Evaluator(torch.nn.Module):
        def predict(self, states):
            policy = torch.zeros((len(states), 100))
            policy[:, 38] = 1
            return policy, torch.zeros((len(states), 1))

    game = SequenceGame()
    game.reset()
    game.board[3, :8] = 1
    game.completed_sequences[1] = [{(3, c) for c in range(5)}]
    game.hands[1] = [Card.from_int(10)]
    search = MCTS(Evaluator(), torch.device("cpu"), temperature=0, dirichlet_epsilon=0)
    move, policy = search._search_python(game, 1, 4)
    assert (move.row, move.col) == (3, 8)
    assert policy.shape == (100,) and policy.sum() == 1
