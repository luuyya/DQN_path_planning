from queue import Queue

class ReplayBuffer():
    def __init__(self,capacity):
        self.buffer=Queue()

    def add(self,state,action,reward,next_state,done):
        pass