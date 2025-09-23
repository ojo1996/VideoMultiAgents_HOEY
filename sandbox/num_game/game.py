import random  
  
def guessing_game():  
    """A simple random number guessing game."""  
    print("Welcome to the Number Guessing Game!")  
    print("I'm thinking of a number between 1 and 100.")  
  
    secret_number = random.randint(1, 100)  
    guess = 0  
  
    while guess != secret_number:  
        try:  
            guess_input = input("What's your guess? ")  
            guess = int(guess_input)  
  
            if guess < secret_number:  
                print("Too low!")  
            elif guess > secret_number:  
                print("Too high!")  
            else:  
                print(f"You got it! The number was {secret_number}.")  
        except ValueError:  
            print("Invalid input. Please enter a number.")  
  
if __name__ == "__main__":  
    guessing_game() 
