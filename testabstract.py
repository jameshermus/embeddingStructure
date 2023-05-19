from abc import ABC, abstractmethod

class testabstract(ABC):
    def __init__(self, value):
        self.value = value

    @abstractmethod
    def implTestMethod1(self):
        pass
    
    @abstractmethod
    def implTestMethod2(self):
        pass

class dog(testabstract):
    def __init__(self,value):
        super().__init__(value)
        pass

    def implTestMethod1(self):
        self.x = 1
        pass

    def implTestMethod2(self):
        self.y = 2
        pass

blaze = dog(1)
