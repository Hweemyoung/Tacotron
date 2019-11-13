from torch import nn

class Taco2Loss(nn.Module):
    def __init__(self):
        super(Taco2Loss, self).__init__()

    def forward(self, preds, targets):
        """
        :param preds: Tuple
            (spectrogram_pred, stop_tokens_cum)
        :param targets: Tuple
            (spectrogram_label, stop_tokens_label)
        :return:
        """
        (spectrogram_pred, spectrogram_length_pred) = preds
        (spectrogram_label, spectrogram_length_label) = targets

        spectrogram_label.requires_grad = False
        spectrogram_length_label.requires_grad = False

        loss_spectrogram = nn.MSELoss()(spectrogram_pred, spectrogram_label)
        loss_length = nn.BCEWithLogitsLoss()(spectrogram_length_pred, spectrogram_length_label)

        return loss_spectrogram + loss_length

