import numpy as np
import numpy.typing as npt


def mask_data(
    x: npt.NDArray, mask_cond: npt.NDArray, mask_iso: bool = True
) -> npt.NDArray:
    """
    Mask values in x based on mask_cond. If there are any isolated
    points (ones with no unmasked neighboring points), mask those as
    well for visual purposes when making line plots. Masked values are
    returned as np.nan.

    """
    x_masked = np.ma.masked_where(mask_cond, x)

    # Return x_masked if nothing or everything is masked
    x_mask = x_masked.mask
    if np.all(x_mask) or np.all(~x_mask):
        return np.ma.filled(x_masked, np.nan)

    if mask_iso:
        if not x_mask[0] and x_mask[1]:
            x_mask[0] = True

        if x_mask[-2] and not x_mask[-1]:
            x_mask[-1] = True

        for start_idx in range(0, len(x_masked) - 3):
            end_idx = start_idx + 3
            window = x_mask[start_idx:end_idx]
            if list(window) == [True, False, True]:
                x_mask[start_idx + 1] = True

    return np.ma.filled(x_masked, np.nan)
