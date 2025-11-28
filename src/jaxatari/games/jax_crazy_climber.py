from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import chex

from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
import jaxatari.spaces as spaces



class CrazyClimberConstants(NamedTuple):
    # Auflösung später für Renderer
    WIDTH: int = 160
    HEIGHT: int = 210

    # Fenster-Grid
    GRID_COLS: int = 6
    GRID_ROWS: int = 64
    VISIBLE_ROWS: int = 12       # wie viel davon im Bild sichtbar ist

    WINDOW_WIDTH: int = 16
    WINDOW_HEIGHT: int = 12

    WINDOW_X_OFFSET: int = 20
    WINDOW_Y_OFFSET: int = 40

    WINDOW_X_SPACING: int = 16
    WINDOW_Y_SPACING: int = 12

    # fenster zustände (-1 empty, 0 open, 1 closed, 2 partial erstmal)
    EMPTY   = -1
    OPEN    = 0
    CLOSED  = 1
    PARTIAL = 2








class JaxCrazyClimber(
    JaxEnvironment[CrazyClimberState, CrazyClimberObservation, CrazyClimberInfo, CrazyClimberConstants]
):
    def __init__(self, consts: CrazyClimberConstants = None):
        consts = consts or CrazyClimberConstants()
        super().__init__(consts)

        # basisgebäude layout jax array
        #self.base_layout = create_base_building_layout(self.consts)


        #self.action_set = [Action.NOOP]

