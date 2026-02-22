# BotBattle - Getting Started

- URL: https://botbattle.be/docs/tutorial-tronk-tronkscript
- Fetched at (UTC): 2026-02-20T21:41:55.818631+00:00

## Extracted Content

BotBattle
Info
Docs
Tournaments
Rankings
Games
Livestream
Editor
Getting Started

Let's get you up and running!

Welcome to the exciting world of the BotBattle! In this tutorial, we will embark on a playful journey where you will learn the art of bot creation and get your system up and running. Get ready to unleash your coding prowess and dive into the thrilling realm of competition. So, buckle up and let's embark on this exhilarating adventure together!

The Boring Stuff

To be able to develop your very own bot that plays Tronk, we've created an online editor that you can use to create bots and play games all from your browser. You can edit the code of all 6 bots on the field, step through their execution and log important information to debug your code. Additionally, the code you write will automatically be saved with your account if you're logged in.

So go ahead and open the web editor using the button below in another tab or window, so you can experiment with the code fragments in this tutorial while you're reading through it!

  Open Web Editor

About Tronkscript

The default code that you see popping up in the web editor may seem quite complicated and intimidating. But don't worry, it's all not as bad as it seems. That said, you'll need to have some background knowledge to help you feel more at ease. So let's dive in for a quick tour!

Tronkscript is a JavaScript simulated programming language, the documentation for which can be found here. A program written in Tronkscript is made up of expressions that are executed one after eachother. Each expression takes a certain number of game ticks or time to be finished. Having lots of time-consuming expressions in your code will slow down your bot!

The complete documentation of Tronkscript can be found by clicking the button below:

  Open Tronkscript Docs

Let's move

Let's start with moving around on the game field. As a first demonstration, we will run to the other side of the field.

  Just like in most programming languages, Tronkscript is case sensitive. Tronkscript uses lines to separate expressions, and does not depend on the space at the start of a line, so no need to worry about indentation! You can format your code however you like. That said, we recommend to use indentation to make your code more readable.

Tronkscript will start executing from the top of the file, and it will continue to execute expressions one after each other until it reaches the end of the file. To keep everything organised, we will define a function called main, which will be the entry point of our application. To execute the main function, we will call it at the end of the file.

Our bot always starts facing the center of the field. In order to let our bot run to the other side, we need to keep the bot alive. This is done by keeping the code busy.


function main() do
    while (1 == 1) do
        -- keep alive
    end while
end function

main()




If you press the play button in the web editor, you'll see that our bot is doing what we expect it to do! That said, it dies when it reaches the edge of the field, because it keeps trying to move forward, but it can't.

Turning Around

Now that we can move forward, let's learn how to turn our bot. Tronkscript provides three simple functions to change direction: turnLeft(), turnRight(), and turnForward().

These functions change the direction of the next step that your bot takes, without moving the bot immediately.

turnLeft() - Rotates your bot 60 degrees to the left
turnRight() - Rotates your bot 60 degrees to the right
turnForward() - Rotates your bot back to the same direction as the previous move

Let's create a simple bot that turns left once and then stays alive:


function main() do
    -- Turn left once
    turnLeft()
    
    -- Keep the bot alive
    while (1 == 1) do
        -- keep alive
    end while
end function

main()




If you run this in the web editor, you'll see that the bot turns 60 degrees to the left and then stays in place.

  Calling turnLeft(), turnRight() or turnForward() multiple times in succession will only apply the last turn. The turn functions simply set the direction your bot will face during the next move. Since the game moves bots forward every 10,000 ticks, turning multiple times before your bot actually moves will have no effect—only the final turn direction will matter.
Timing is Everything

Now that we know how to move and turn, let's learn how to control the timing of our bot's actions. The key to controlling timing in Tronkscript is the wait() and getTick() functions.

The wait(n) function pauses execution for n ticks. This allows us to slow down our bot's actions and create more deliberate, controlled movements. It's also useful for synchronizing your bot's actions with the game's timing.

Let's create a simple bot that moves in circles. Each time setting the turn, then waiting for the next movement, and returning.



function waitForNextMove() do
	currentTick = getTick()
	ticksSinceLastMove = currentTick % 10000
	ticksUntilNextMove = 10000 - ticksSinceLastMove
	wait(ticksUntilNextMove)
end function

function main() do
    -- First turn left, then wait for the next movement
	turnLeft()
	waitForNextMove()

	-- Now create a loop of turning right and waiting to create a circular movement pattern
	while (1 == 1) do
	    turnRight()
	    waitForNextMove()
	end while
end function

main()



As you can see, the wait() function takes a single argument: the number of ticks to wait. You can use any number you like, allowing you to create different timing patterns for your bot.

  Keep in mind that the calculations inside the waitForNextMove() function (like calling getTick() and performing arithmetic) also consume ticks. This means the actual timing of when your bot moves will be off by a few ticks compared to the ideal 10,000 tick interval. For most purposes this is fine, but if you need precise timing, you may need to account for these extra ticks!
Reading the Field

Now that we can move and turn, let's learn how to observe our surroundings. Tronkscript provides two functions to check what's on specific tiles: getTileRel(q, r) and getTileAbs(q, r).

getTileRel(q, r) checks tiles relative to your bot's current position using cube coordinates (q, r). For example:

getTileRel(0, -1) - The tile just in front of your bot
getTileRel(1, -1) - The tile to the right
getTileRel(1, 0) - The tile to the left

These functions return a tuple with four values: (exists, isEmpty, playerId, isGem). Let's use these to create a bot that searches for gems and picks them up!


function waitForNextMove() do
	currentTick = getTick()
	ticksSinceLastMove = currentTick % 10000
	ticksUntilNextMove = 10000 - ticksSinceLastMove
	wait(ticksUntilNextMove)
end function

function checkAndPickupGem(q, r) do
	exists, isEmpty, playerId, isGem = getTileRel(q, r)
	
	-- Check if the tile exists
	if (exists == 0) then
		return 0 -- Tile doesn't exist
	end if
	
	-- Check if there's a gem
	if (isGem == 1) then
		return 1 -- Gem found!
	end if
	
	return 0 -- No gem here
end function

function main() do
	while (1 == 1) do
		-- Check in front first
		temp = checkAndPickupGem(0, -1)
		if (temp == 1) then
			turnForward()
		end if
		
		-- Check to the right
		temp = checkAndPickupGem(1, -1)
		if (temp == 1) then
			turnRight()
		end if
		
		-- Check to the left
		temp = checkAndPickupGem(-1, 0)
		if (temp == 1) then
			turnLeft()
		end if
		
		-- If no gem found, now just wait for the next move
		waitForNextMove()
	end while
end function

main()



In this example, we check for gems in three directions: front, right, and left. If we find a gem, we move towards it. If not, we turn and wait for the next movement tick.

  When a tile doesn't contain a player, the playerId value will be -1. You can use this to detect if a player is present on a tile.

Copyright © BotBattle FV 2023-2026
