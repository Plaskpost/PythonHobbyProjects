import pygame

pygame.init()

# Create a Pygame window
window = pygame.display.set_mode((400, 400))

# Dictionary to dynamically map key values to key constant names
key_constant_map = {}

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key = event.key
            key_name = pygame.key.name(key)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_KP7]:
                print("worked")

            if key not in key_constant_map:
                key_constant_name = "K_" + key_name.upper()
                key_constant_map[key] = key_constant_name

            key_constant_name = key_constant_map[key]
            print("Key pressed:", key_name)
            print("Key constant:", key_constant_name)

pygame.quit()
