import pygame
import sys
import random


# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simple RPG")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
ROAD_COLOR = (100, 100, 100)  # Dark gray for roads

# Player
player_pos = [WIDTH // 2, HEIGHT // 2]
player_radius = 20
player_speed = 10 #FIXME

# Areas
house = pygame.Rect(50, 250, 100, 100)
work = pygame.Rect(650, 50, 100, 100)
library = pygame.Rect(650, 450, 100, 100)
shop = pygame.Rect(50, 50, 100, 100)

# Game state
money = 0
intelligence = 0
actions_left = 16
current_location = "outside"
action_cooldown = 0
promotion_available = False
house_upgraded = False

# Work tiers
work_tiers = [
    {"name": "Clerk", "salary": 5, "intelligence_required": 0},
    {"name": "Manager", "salary": 10, "intelligence_required": 20},
    {"name": "Director", "salary": 20, "intelligence_required": 50},
    {"name": "Vice President", "salary": 50, "intelligence_required": 100},
    {"name": "CEO", "salary": 100, "intelligence_required": 200}
]

current_tier = 0

# Font
font = pygame.font.Font(None, 36)

def draw_player():
    pygame.draw.circle(screen, RED, player_pos, player_radius)

def draw_areas():
    pygame.draw.rect(screen, GREEN, house)
    pygame.draw.rect(screen, BLUE, work)
    pygame.draw.rect(screen, YELLOW, library)
    pygame.draw.rect(screen, GRAY, shop)

def draw_roads():
    # Vertical road
    pygame.draw.rect(screen, ROAD_COLOR, (WIDTH // 2 - 10, 0, 20, HEIGHT))
        
    # Connection to house
    pygame.draw.rect(screen, ROAD_COLOR, (house.right, house.centery - 10, WIDTH // 2 - house.right, 20))
    
    # Connection to shop
    pygame.draw.rect(screen, ROAD_COLOR, (shop.right, shop.centery - 10, WIDTH // 2 - shop.right, 20))
    
    # Connection to work
    pygame.draw.rect(screen, ROAD_COLOR, (WIDTH // 2, work.centery - 10, work.left - WIDTH // 2, 20))
    
    # Connection to library
    pygame.draw.rect(screen, ROAD_COLOR, (WIDTH // 2, library.centery - 10, library.left - WIDTH // 2, 20))

def draw_ui():
    money_text = font.render(f"Money: ${money}", True, WHITE)
    intelligence_text = font.render(f"Intelligence: {intelligence}", True, WHITE)
    actions_text = font.render(f"Actions left: {actions_left}", True, WHITE)
    location_text = font.render(f"Location: {current_location}", True, WHITE)
    tier_text = font.render(f"Job: {work_tiers[current_tier]['name']}", True, WHITE)
    
    screen.blit(money_text, (10, 10))
    screen.blit(intelligence_text, (10, 50))
    screen.blit(actions_text, (10, 90))
    screen.blit(location_text, (10, 130))
    screen.blit(tier_text, (10, 170))

    if promotion_available:
        promotion_text = font.render("Press 'P' for promotion!", True, WHITE)
        screen.blit(promotion_text, (WIDTH // 2 - 100, 10))

def check_promotion():
    global promotion_available
    for i, tier in enumerate(work_tiers):
        if intelligence >= tier["intelligence_required"] and i > current_tier:
            promotion_available = True
            return

def show_shop_menu():
    menu_rect = pygame.Rect(WIDTH // 4, HEIGHT // 4, WIDTH // 2, HEIGHT // 2)
    pygame.draw.rect(screen, WHITE, menu_rect)
    
    title_text = font.render("Shop", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - 30, HEIGHT // 4 + 10))

    if not house_upgraded:
        house_text = font.render("1. Upgrade House ($10,000)", True, BLACK)
        screen.blit(house_text, (WIDTH // 4 + 10, HEIGHT // 4 + 50))
    
    exit_text = font.render("2. Exit Shop", True, BLACK)
    screen.blit(exit_text, (WIDTH // 4 + 10, HEIGHT // 4 + 90))


class GuessingGame:
    def __init__(self):
        self.target = random.randint(1, 100)
        self.attempts = 0
        self.max_attempts = 7
        self.guess = ""
        self.message = "Guess a number between 1 and 100"
        self.game_over = False
        self.bonus_points = 0

    def make_guess(self, guess):
        if self.game_over:
            return

        self.attempts += 1
        guess = int(guess)

        if guess < self.target:
            self.message = "Too low!"
        elif guess > self.target:
            self.message = "Too high!"
        else:
            self.message = f"Correct! You guessed it in {self.attempts} attempts!"
            self.bonus_points = self.max_attempts - self.attempts + 1
            self.game_over = True

        if self.attempts >= self.max_attempts and not self.game_over:
            self.message = f"Game over! The number was {self.target}"
            self.game_over = True

    def draw(self, screen, font):
        # Draw background
        pygame.draw.rect(screen, (200, 200, 200), (100, 100, 600, 400))
        
        # Draw title
        title = font.render("Guess the Number", True, (0, 0, 0))
        screen.blit(title, (350 - title.get_width() // 2, 120))

        # Draw message
        message = font.render(self.message, True, (0, 0, 0))
        screen.blit(message, (400 - message.get_width() // 2, 200))

        # Draw input box
        pygame.draw.rect(screen, (255, 255, 255), (300, 250, 200, 40))
        guess_surface = font.render(self.guess, True, (0, 0, 0))
        screen.blit(guess_surface, (310, 260))

        # Draw attempts
        attempts = font.render(f"Attempts: {self.attempts}/{self.max_attempts}", True, (0, 0, 0))
        screen.blit(attempts, (350, 300))

        if self.game_over:
            if self.bonus_points > 0:
                bonus = font.render(f"Bonus points: {self.bonus_points}", True, (0, 128, 0))
                screen.blit(bonus, (350, 340))
            
            replay = font.render("Press SPACE to play again", True, (0, 0, 0))
            screen.blit(replay, (300, 400))


def main():
    global player_pos, money, intelligence, actions_left, current_location, action_cooldown, promotion_available, current_tier, house_upgraded

    clock = pygame.time.Clock()
    show_shop = False
    
    # Move these outside the game loop
    guessing_game = None
    playing_minigame = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p and promotion_available and current_location == "work":
                    current_tier += 1
                    promotion_available = False
                if event.key == pygame.K_SPACE and current_location == "shop":
                    show_shop = True
                if show_shop:
                    if event.key == pygame.K_1 and not house_upgraded and money >= 10000:
                        money -= 10000
                        house_upgraded = True
                        show_shop = False
                    elif event.key == pygame.K_2:
                        show_shop = False
                
                # Handle guessing game input
                if playing_minigame:
                    if event.key == pygame.K_RETURN and not guessing_game.game_over:
                        guessing_game.make_guess(guessing_game.guess)
                        guessing_game.guess = ""
                    elif event.key == pygame.K_BACKSPACE:
                        guessing_game.guess = guessing_game.guess[:-1]
                    elif event.key == pygame.K_SPACE and guessing_game.game_over:
                        intelligence += guessing_game.bonus_points
                        actions_left -= 1
                        playing_minigame = False
                    elif event.unicode.isnumeric():
                        guessing_game.guess += event.unicode

        keys = pygame.key.get_pressed()
        
        # Player movement
        if not show_shop and not playing_minigame:  # Prevent movement while shop menu or minigame is open
            if keys[pygame.K_LEFT] and player_pos[0] > player_radius:
                player_pos[0] -= player_speed
            if keys[pygame.K_RIGHT] and player_pos[0] < WIDTH - player_radius:
                player_pos[0] += player_speed
            if keys[pygame.K_UP] and player_pos[1] > player_radius:
                player_pos[1] -= player_speed
            if keys[pygame.K_DOWN] and player_pos[1] < HEIGHT - player_radius:
                player_pos[1] += player_speed

        # Check for area interactions
        player_rect = pygame.Rect(player_pos[0] - player_radius, player_pos[1] - player_radius, player_radius * 2, player_radius * 2)

        if player_rect.colliderect(house):
            current_location = "house"
            if keys[pygame.K_SPACE]:
                actions_left = 20 if house_upgraded else 16
        elif player_rect.colliderect(work):
            current_location = "work"
            if actions_left > 0 and keys[pygame.K_SPACE] and action_cooldown == 0:
                money += work_tiers[current_tier]["salary"]
                actions_left -= 1
                action_cooldown = 15
        elif player_rect.colliderect(library):
            current_location = "library"
            if actions_left > 0 and keys[pygame.K_SPACE] and action_cooldown == 0:
                intelligence += 1
                actions_left -= 1
                action_cooldown = 15
                check_promotion()
            elif keys[pygame.K_m] and not playing_minigame and actions_left > 0:
                playing_minigame = True
                guessing_game = GuessingGame()
        elif player_rect.colliderect(shop):
            current_location = "shop"
        else:
            current_location = "outside"

        if action_cooldown > 0:
            action_cooldown -= 1

        # Drawing
        screen.fill(BLACK)
        draw_roads()  # Draw roads before areas and player
        draw_areas()
        draw_player()
        draw_ui()

        if show_shop:
            show_shop_menu()
        
        if playing_minigame:
            guessing_game.draw(screen, font)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()