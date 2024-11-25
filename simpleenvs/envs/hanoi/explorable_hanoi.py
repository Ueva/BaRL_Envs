from simpleenvs.envs.hanoi import HanoiEnvironment


class ExplorableHanoiEnvironment(HanoiEnvironment):
    def __init__(
        self,
        num_disks=4,
        num_poles=3,
        action_penalty=-0.001,
        goal_reward=1.0,
        start_state=None,
    ):
        super(ExplorableHanoiEnvironment, self).__init__(
            num_disks=num_disks,
            num_poles=num_poles,
            action_penalty=action_penalty,
            goal_reward=goal_reward,
            start_state=start_state,
        )
        self.goal_state = self.num_disks * (-1,)

    def is_state_terminal(self, state=None):
        return False

    def get_available_actions(self, state=None):
        legal_actions = [i for i, action in enumerate(self.action_list) if self._is_action_legal(action, state=state)]
        return legal_actions
