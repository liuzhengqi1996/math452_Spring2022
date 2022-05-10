# This is module

#Define a class
class Math497():
    def __init__(self):
        self.num_students = 40
        self.num = 100
    def add_students(self,x):
        self.num_students = self.num_students + x
        
# Define a child class of Math497()
class undergraduate(Math497):
    def __init__(self):
        self.num_undergraduate_students = 34
        super().__init__()
    def add_undergraduate_students(self,x):
        self.num_undergraduate_students = self.num_undergraduate_students + x
        self.num_students = self.num_students + x
