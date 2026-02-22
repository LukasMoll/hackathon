# BotBattle - Tronk

- URL: https://botbattle.be/docs/tronk
- Fetched at (UTC): 2026-02-20T21:41:56.223385+00:00

## Extracted Content

BotBattle
Info
Docs
Tournaments
Rankings
Games
Livestream
Editor
Tronk

Learn all about this year's game.

Tronk is a custom multiplayer arcade game inspired on the popular game Snake. We tried to keep it as simple and minimalistic as possible. Here's everything you need to know.

Objective

The objective of Tronk is to be the last player standing by strategically walking around on the game field and avoiding other players and picking up gemstones to grow and trap your opponents. That's it.

Preview of the Arena
Gameplay Mechanics
Each player controls a character (those round colorful thingies with cute eyes) on a hexagonal field containing 91 tiles.
Every 10000th tick, players move one tile forward in the direction they are facing, as long as that tile exists and is not occupied by another player.
Moving to a tile occupied by another player or non-existing tile will result in a game-over. If multiple players want to move to the same tile, they also crash.
Moving to a tile occupied by a gemstone (round green thingies on the game field) will result in the player getting one tile longer. A new gemstone will spawn on another empty tile.
At any time, players can set the direction in which they want to move by turning their character.
Players can only turn to face the tile in front, to the left or to the right with respect to the tile they moved from. They thus cannot take sharp turns.
Rules
The game starts with all 6 players placed in the corners on the arena, and 6 gemstones in the locations indicated above.
Each player's bot code is executed simultaneously, not in turns.
Each player has one life. When a player loses this life by moving to a non-empty tile, they are eliminated from the game.
When a player dies, their character remains on the game field.
Every 10000 ticks, all players will move one tile in the direction they are facing simultaneously.
Players can become longer by walking on tiles with gemstones.
When a gemstone is picked up, it is removed from the game. A new gemstone is automatically placed on a random empty tile on the game field.
If no gemstone is picked up after 50000 ticks (5 steps), a new gemstone is added on a random empty tile every 10000 ticks (1 step), until some player picks up a gemstone.
The game continues until only one player remains, who is declared the winner.
Technical Requirements
Participant must submit their bot as a single code file with a maximum size of 1MB. You may upload as many bots as you like, though only one can be active at any time.
Bots may not crash. If the execution of your code file exits, your bot will instantly die.
Example Gameplay

You can take a look at the livestream here or the replay videos of past games to get an idea of how the gameplay of Tronk looks.

You can also play around with your bots in de web editor!

Detailed Description

Below, you'll find some more detailed information about various aspects of the game. You don't necessarily need to know these in order to start writing your bot, but they will be useful if you plan on optimizing your algorithm to get the most out of your bot.

Game field
The game is played on a hexagonal game field with a total of 91 tiles. There are 6 tiles on every edge of the field.
Legal moves
Every 10000 ticks, all players move one tile in the direction they are facing. There are a few scenarios that can occur:
The tile is outside the game field.
If you move to a tile outside the game field, your player is eliminated from the game.
The tile is occupied by a player.
If someone else (or you yourself) is occupying a tile, you cannot move there and your bot will die.
The tile is not occupied by a player and you're the only one moving there.
If you're the only one moving to an empty tile or a tile with a gemstone, your player will step to that tile and continues to live.
The tile is not occupied by a player and there are multiple players moving there.
If two or more players are moving to the same empty tile of tile with a gemstone, they both die.
Gemstones
Gemstones are the apples from the game Snake. If a player moves to a tile with a gemstone on it, the gemstone is removed and the player's character increases its length by one tile. There is no limit on how long a player can get.
Gemstone spawning
At the start of every game, 6 gemstones are placed close to the center (see the locations indicated on the arena preview above). When a gemstone is removed from the game (because some player moved to that tile), a new one is generated automatically and placed on a random empty tile of the game field. Additionally, if no player moves to a tile with a gemstone in 50000 ticks (5 steps), a new gemstone is added on a random empty tile every 10000 ticks (1 step). This continues until a player picks up a gemstone, after which this timer is reset.
Tick system
The game runs on a tick-based system. Each tick, every living player's bot code is advanced by one step. Players physically move once every 10,000 ticks. This means your bot has a budget of 10,000 ticks between moves to decide which direction to turn.

Every function your bot calls has a tick cost. For example, calling turnLeft() costs 100 ticks, meaning your bot is paused for 100 ticks while the turn is processed. Regular statements (assignments, conditions, loops) each cost 1 tick. If your bot uses up all 10,000 ticks before the next move, it simply moves in whatever direction it was last set to face.

Tick costs of game functions:
turnLeft(), turnRight(), turnForward(), turn(dir): 100 ticks
getPlayerInfo(id): 50 ticks
getTileAbs(q, r), getTileRel(q, r): 20 ticks
getTurn(): 10 ticks
relToAbs(q, r): 5 ticks
getPlayerId(), getTick() : 1 tick
Simultaneous execution
All players' bots are ticked in order (player 0 through 5) each game tick, but movement happens simultaneously. When the 10,000-tick boundary is reached, all players' moves are validated at once before any player actually moves. This means two players moving toward the same empty tile will both die. Neither has priority.
Game-specific Functions

Here you find an extended description of the game-specific functions available in Tronk.

### getPlayerId
Returns the ID of your player in the current game. Takes no arguments. This ID is a number from 0 to 5, corresponding to the index in the array of players in the game.

Arguments
This function does not take any arguments.

Examples
Example	Description
id = getPlayerId()
id is between 0 and 5
Notes
This function takes 1 tick to complete.
### turnLeft
Sets the direction of the player so it faces left. Calling this function twice will not further rotate the player, as a player can only move left, right or forward with respect to the previous tile.

Arguments
This function does not take any arguments.

Examples
Example	Description
turnLeft()
player will face forward direction
Notes
This function takes 100 ticks to complete.
### turnRight
Sets the direction of the player so it faces right. Calling this function twice will not further rotate the player, as a player can only move left, right or forward with respect to the previous tile.

Arguments
This function does not take any arguments.

Examples
Example	Description
turnRight()
player will face forward direction
Notes
This function takes 100 ticks to complete.
### turnForward
Sets the direction of the player so it faces forward. Calling this function twice will not further rotate the player, as a player can only move left, right or forward with respect to the previous tile.

Arguments
This function does not take any arguments.

Examples
Example	Description
turnForward()
player will face forward direction
Notes
This function takes 100 ticks to complete.
### turn
Sets the direction of the player so it faces the specified direction. The argument is -1 for left, 0 for forward, or 1 for right. Calling this function twice will not further rotate the player, as a player can only move left, right or forward with respect to the previous tile.

Arguments
Argument	Type
Direction to turn to	The argument can be a number or a variable
Examples
Example	Description
turn(-1)
player will face left
turn(0)
player will face forward
turn(1)
player will face right
turn(5)
argument should be -1, 0 or 1
Notes
This function takes 100 ticks to complete.
### getTurn
Returns the direction of the player that is currently set with one of the turn functions. The return value is -1 for left, 0 for forward and 1 for right. Takes zero arguments.

Arguments
This function does not take any arguments.

Examples
Example	Description
dir = getTurn()
dir is -1 for left, 0 for forward and 1 for right
Notes
This function takes 10 ticks to complete.
### getTileAbs
Returns info about a tile on the game field. The specified coordinates are in the absolute coordinate system, with the center of the field at the origin. Return data is an array with 4 elements. First element is 1 is the tile exists on the game field and 0 if it is out of the field bounds. Second element is 1 is the tile is empty and 0 if it is occupied by a player or gemstone. Third element equals the ID of the player that is on the tile, or -1 if no player is on the tile. Fourth element is 1 if a gemstone is on the tile and 0 if no gemstone is on the tile.

Arguments
Argument	Type
Absolute q coordinate of tile	The argument can be a number or a variable
Absolute r coordinate of tile	The argument can be a number or a variable
Examples
Example	Description
q = 1
r = 2
exists, isEmpty, playerId, hasGemstone = getTile(q, r)
gets info about tile (q, r) = (1, 2)
q = 99
r = 0
exists, _, _, _ = getTile(q, r)
exists will be 0 because tile is out of the game field
_, isEmpty, _, _ = getTile(0, 0)
isEmpty will be 1 is nothing occupies the tile at the origin
_, _, playerId, _ = getTile(0, 0)
playerId will be the ID of the player that is occupying the tile at the origin (from 0 to 5), or -1 if there isn't any
_, _, _, hasGemstone = getTile(0, 0)
hasGemstone will be 1 if there is a gemstone at the origin, 0 otherwise
Notes
This function takes 20 ticks to complete.
### getTileRel
Returns info about a tile on the game field. The specified coordinates are relative to the player, with the center of the field at the player's position and rotated with the player's facing direction. The Q-axis lies to the right of the direction that the player is facing. The R-axis lies to the back left direction relative to the player's orientation. Return data is an array with 4 elements. First element is 1 is the tile exists on the game field and 0 if it is out of the field bounds. Second element is 1 is the tile is empty and 0 if it is occupied by a player or gemstone. Third element equals the ID of the player that is on the tile, or -1 if no player is on the tile. Fourth element is 1 if a gemstone is on the tile and 0 if no gemstone is on the tile.

Arguments
Argument	Type
Relative q coordinate of tile	The argument can be a number or a variable
Relative r coordinate of tile	The argument can be a number or a variable
Examples
Example	Description
q = 0
r = -1
exists, isEmpty, playerId, hasGemstone = getTile(q, r)
gets info about tile in front of the player
q = 1
r = -1
exists, isEmpty, playerId, hasGemstone = getTile(q, r)
gets info about tile to the front right of the player
q = -1
r = 0
exists, isEmpty, playerId, hasGemstone = getTile(q, r)
gets info about tile to the front left of the player
q = 99
r = 0
exists, _, _, _ = getTile(q, r)
exists will be 0 because tile is out of the game field
_, isEmpty, _, _ = getTile(0, -1)
isEmpty will be 1 is nothing occupies the tile in front of the player
_, _, playerId, _ = getTile(0, -1)
playerId will be the ID of the player that is occupying the tile in front of the player (from 0 to 5), or -1 if there isn't any
_, _, _, hasGemstone = getTile(0, -1)
hasGemstone will be 1 if there is a gemstone at the tile in front of the player, 0 otherwise
Notes
This function takes 20 ticks to complete.
### relToAbs
The specified coordinates are relative to the player, with the center of the field at the player's position and rotated with the player's facing direction. The Q-axis lies to the right of the direction that the player is facing. The R-axis lies to the back left direction relative to the player's orientation.

Arguments
Argument	Type
Relative q coordinate of tile	The argument can be a number or a variable
Relative r coordinate of tile	The argument can be a number or a variable
Examples
Example	Description
q = 0
r = 0
q_abs, r_abs = relToAbs(q, r)
q_abs and r_abs will be the position of the player in absolute coordinates
Notes
This function takes 5 ticks to complete.
### getTick
Returns the current tick of the game. See the game documentation for more information about ticks and how time is managed. Takes zero arguments.

Arguments
This function does not take any arguments.

Examples
Example	Description
now = getTick()
now will be set to the current game tick
Notes
This function takes 1 tick to complete.
### getPlayerInfo
Returns info about a player in the game. The specified ID is equal to the index of the player in the array of all players (from 0 to 5). Return data is an array with 4 elements. First element is 1 is the tile exists on the game field and 0 if it is out of the field bounds. Second element is 1 is the tile is empty and 0 if it is occupied by a player or gemstone. Third element equals the ID of the player that is on the tile, or -1 if no player is on the tile. Fourth element is 1 if a gemstone is on the tile and 0 if no gemstone is on the tile.

Arguments
Argument	Type
ID of the player	The argument can be a number or a variable
Examples
Example	Description
id = 0
alive, headQ, headR, headFacing, length = getPlayerInfo(id)
gets info about the first player of the internal game state
id = getPlayerId()
alive, headQ, headR, headFacing, length = getPlayerInfo(id)
gets info about the current player
alive, _, _, _, _ = getPlayerInfo(0)
alive is 1 if the first player is alive, otherwise 0
_, headQ, headR, _, _ = getPlayerInfo(0)
headQ and headR indicate the position of the player's head in absolute coordinates
_, _, _, headFacing, _ = getPlayerInfo(0)
headFacing is 0 when the player's head is facing up, 1 for up right, 2 for down right, 3 for down, 4 for down left and 5 for up left
_, _, _, _, length = getPlayerInfo(0)
length indicates how long the player's body is
Notes
This function takes 50 ticks to complete.

Copyright © BotBattle FV 2023-2026
