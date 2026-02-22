# BotBattle - Rulebook

- URL: https://botbattle.be/rulebook
- Fetched at (UTC): 2026-02-20T21:41:55.445981+00:00

## Extracted Content

BotBattle
Info
Docs
Tournaments
Rankings
Games
Livestream
Editor
Sign In
The Rulebook

Wanna win? Gotta play by the rules!

The Rulebook was last updated on 10/10/2024.

1. Introduction

The BotBattle is a monthlong online programming competition where participants write a small program to play a multiplayer arcade game against the programs of other participants. These AIs or so-called bots can view the game area and must efficiently implement a strategy to move the character around and outwin their opponents.

The BotBattle is organised by BotBattle FV in collaboration with CenEka, the official student association for Computer Science Engineering students and Electrical Engineering students at Ghent University. The purpose of the BotBattle competition is multifaceted. In addition to providing an enjoyable experience for participating students, the competition aims to achieve the following objectives:
Student Engagement: The competition seeks to actively engage students by offering them an exciting platform to showcase their programming skills and strategic thinking abilities.
Event Promotion: The competition serves as a means to generate publicity and raise awareness for the events organised by BotBattle partners. By attracting more students to participate, it contributes to the overall success and visibility of their activities.

This Rulebook outlines the guidelines, regulations, and instructions for participants to follow during the competition. It serves as a reference for participants to understand the rules of the competition, the objectives, the allowed actions, and any limitations or restrictions that may apply.

2. Eligibility

To participate in the BotBattle, you'll need to have a valid email address from an institution of AUGent (ending in @ugent.be, @hogent.be, @arteveldehs.be or @howest.be). We need to make sure each participant can only create one account as to prevent malicious manipulation of the participant's ranking. If you don't have access to such an email address, you can request an exception by contacting us, see below.

All bachelor students, master students, PhD students and AUGent staff can create a BotBattle account and submit bots that will take part in the competition. That said, to be eligible for prizes, participants must officially be enrolled as 1) a student at Ghent University for at least 27 credits (with the exception of their graduation year) in a bachelor or master program, or 2) a PhD-student at an institution of the Association Ghent University at the time of registration for the competition.

Participants are required to register individually for the competition by creating a BotBattle account through the website. After registration, participants can submit their self-developed bots. Bots should adhere to the guidelines and specifications provided for the game, see section 3.

While participants register and submit bots individually, they are encouraged to form teams with other participants and engage in strategy discussions. Collaborative efforts can enhance the overall experience and foster a sense of community within the competition.

Participants are allowed to team up with others during the actual gameplay. This means that they can collaborate with other participants in real-time to strategize and coordinate their actions. However, it is important to note that the prizes awarded for the competition cannot be split among multiple participants.

3. Bot Submission

Participants can submit their bot using the website after signing in with their BotBattle account. The number of submissions in unlimited, though only one submission can be active at any given moment. The participants can choose which bot is active through the website.

A submission consist of one file containing the code of the bot. The maximum size of this file is 1MB. More information about this file and concrete code examples can be found on the Getting Started page.

When creating a new submission, the website analyses the uploaded file to check if it is a valid bot. If a submission is not accepted, the participant will see a message stating the reason for it's rejection.

After a participant has submitted their first bot, they become active on the leaderboard and they will start competing in the games.

All uploaded bots receive a unique identifier (a so-called Bot ID) used internally by the system. Participants should not share these Bot IDs with others but might be asked by officials to provide these for debugging purposes.

4. Evaluation & Ranking

During the competition, the system will continuously pick 6 active bots from the leaderboard and let them play a game of Bonk against eachother. The matching algorithm used for this is based on the participant's rating and makes sure every participant will be chosen with an equal probability over long periods of time.

The outcome of the game is used to update the rating of the participants. Only the relative ranking of a round is preserved, not the absolute length of the period during which a bot was alive.

The rating of all 6 participants is updated through an implementation of a multiplayer ELO rating system. The higher the rating, the better the participant's bot. This calculation takes into account the relative rating of the participants at the start of the game when calculating the change. If a bot with a high rating loses against a bot with a lower rating, its rating will drop more than if it were to lose against a bot with the same rating for example.

A participant starts with a rating of 1000. Due to the mathematical details of the ELO system, this will always be the average rating of all participants.

This ELO rating system only applies to the main leaderboard. The ranking system of the special prize categories is based on counters that increase every time you get a specific achievement. To encourage players to continuously improve their bots and to give new participants a fair chance in winning a special prize, the ranking of these special prize categories is only based on games that were played in the last 72 hours. Details about this can be found in section 5.

The ratings of all participants are published on the competition's website. The outcome of each game is also visible. That being said, we'll never display or communicate your real name with others. Your ranking and bot performance will be linked to a randomly generated username. We'll only reveal your identity if you receive a prize on the closing ceremony.

72 hours before the end of the competition, the leaderboards will be frozen. Though the updates are not visible to the public, games are still played and ratings continue to be updated internally. This measure is taken to keep the last days of the competition even more thrilling and make the closing ceremony more surprising. After the end of the competition, all game history is published to the website so participants can check the validity of the final rankings.

We want to ensure a fair and transparent environment for all participants in this online programming competition. To maintain the integrity of the competition and uphold the standards, we reserve the right to disqualify any participant if deemed necessary.

5. Prizes

At the end of the competition, some of the participants will receive prizes for their achievements. The list below contains the criteria to receive the prize for each of the prize categories.

5.1 Prizes for Students

The prizes in this subsection are only available for students.

Winner of BotBattle 2024
The participant who has the highest rating at the end of the competition. Participants are sorted by their ELO rating in descending order. In case of a tie, the participants with equal ratings are sorted once more by the total number of games their bots played in descending order. If there still is a tie, the participants with equal position are permuted randomly. From this sorted list, the first participant will receive this prize.
BotBattle 2024 Runner-up
The participant who has the second highest rating at the end of the competition. Participants are sorted by their ELO rating in descending order. In case of a tie, the participants with equal ratings are sorted once more by the total number of games their bots played in descending order. If there still is a tie, the participants with equal position are permuted randomly. From this sorted list, the second participant will receive this prize.
BotBattle 2024 Third Place
The participant who has the third highest rating at the end of the competition. Participants are sorted by their ELO rating in descending order. In case of a tie, the participants with equal ratings are sorted once more by the total number of games their bots played in descending order. If there still is a tie, the participants with equal position are permuted randomly. From this sorted list, the third participant will receive this prize.
Special Prize Category 1
Crash Test Dummy
The participant who has bonked into the most walls. For every time your bot walks into a wall, we'll add 1 to your counter for that game. The average of this counter per game over the last 72 hours is the ranking for your bot in this special prize category. The player with the highest ranking at the end of the competition wins this prize. In case of a tie, the participant with the most number of games played during the last 72 hours of the competition wins the prize. If there's still a tie, we'll choose one of the players at random.
Special Prize Category 2
Pillar of Patience
The participant who has stayed on center tile for the longest time. For every tick in the game that your bot is located at the center tile of the game field, we'll add 1 to your counter for that game. The average of this counter per game over the last 72 hours is the ranking for your bot in this special prize category. The player with the highest ranking at the end of the competition wins this prize. In case of a tie, the participant with the most number of games played during the last 72 hours of the competition wins the prize. If there's still a tie, we'll choose one of the players at random.
Special Prize Category 3
Headless Chicken
The participant who has moved to the most number of different tiles. For every distinct tile that your bot moves to, we'll add 1 to your counter for that game. The average of this counter per game over the last 72 hours is the ranking for your bot in this special prize category. The player with the highest ranking at the end of the competition wins this prize. In case of a tie, the participant with the most number of games played during the last 72 hours of the competition wins the prize. If there's still a tie, we'll choose one of the players at random.
5.2 Prizes for Students

The prizes in this subsection are only available for PhD-students.

Top PhD-students
The participants with the highest ratings at the end of the competition. Participants are sorted by their ELO rating in descending order. In case of a tie, the participants with equal ratings are sorted once more by the total number of games their bots played in descending order. If there still is a tie, the participants with equal position are permuted randomly. From this sorted list, the first [TBD] participants will receive this prize.

Participants are allowed to receive multiple prizes, though the maximum number any single person can receive is limited to two.

Participants will receive their prizes at the Closing Ceremony on 05/11 in the Therminal.If a winner is not present at the event, their prize will be handed out to the next participant in the ranking.Contact us if you won't be able to make it to the venue due to unforeseen circumstances. The actual prizes are listed on this page. The prizes will be handed out in the following order:

BotBattle 2024 Third Place
BotBattle 2024 Runner-up
Winner of BotBattle 2024
Special Prize Categories:
Crash Test Dummy
Pillar of Patience
Headless Chicken
Top PhD-students

For the special prize categories, winners can pick their preferred prize from the remaining prizes. This means that the winner of the first category will have full choice, while the following winners can only choose from the prizes that are still available.

6. Intellectual Property

Intellectual Property Rights (IPR) are an important aspect to consider in any creative endeavor, including online programming competitions. We want to ensure that participants in this competition understand their rights and have clarity regarding the ownership of their software submissions.

By participating in this online programming competition and submitting your bot for evaluation, you retain full ownership of your software. This means that you continue to hold the rights to your creation, even after uploading it to the competition system.

We respect and acknowledge the value of your intellectual property. Our intention is solely to evaluate and assess the performance of the submitted bots within the context of the competition. We will not claim ownership or any rights over your software.

However, it is important to note that by submitting your bot, you grant us a non-exclusive license to use, reproduce, modify, and distribute your software solely for the purpose of evaluating and conducting the competition. This license is limited to the duration of the competition and does not extend beyond that.

We will take reasonable measures to protect the confidentiality and security of your software submissions. However, it is advisable to avoid including any proprietary or sensitive information in your code that you do not wish to share with others.

We encourage participants to respect the intellectual property rights of others and refrain from using copyrighted or patented materials without proper authorization. Any infringement of third-party intellectual property rights is the sole responsibility of the participant, and we will not be held liable for any such violations.

7. Fair Play

Fair play is a fundamental principle that underpins the integrity and spirit of any competition, including this online programming competition. We strive to create a level playing field for all participants and ensure that the competition is conducted in a fair and transparent manner.

Original Work: Participants are expected to submit their own original work. Plagiarism, unauthorized use of others' code, or any form of cheating is strictly prohibited. All submissions should be the result of the participant's own efforts and creativity.
Respect for Rules: Participants must adhere to the rules and guidelines outlined in the competition rulebook. Any violation of these rules may result in disqualification or other appropriate actions as determined by the competition organizers.
Prohibited Activities: Participants should refrain from engaging in any activities that may compromise the fairness of the competition. This includes but is not limited to hacking, exploiting vulnerabilities, or attempting to disrupt the competition platform or other participants' submissions.
Collaboration and Communication: While collaboration and discussion among participants are encouraged, it is important to maintain fairness. Sharing code or solutions that give an unfair advantage to others is not allowed. Participants should respect the spirit of competition and avoid sharing sensitive information or strategies that may undermine fair play.
Sportsmanship: Participants are expected to display good sportsmanship throughout the competition. This includes treating fellow participants, organizers, and staff with respect and courtesy. Any form of harassment, discrimination, or disrespectful behavior will not be tolerated.
Compliance with Laws and Regulations: Participants must ensure that their submissions comply with all applicable laws and regulations, including intellectual property rights, data protection, and privacy laws. Any violation of legal requirements may lead to disqualification and potential legal consequences.
Reporting Violations: If participants become aware of any violations of fair play or suspect any misconduct, they should promptly report it to the competition organizers. This helps maintain the integrity of the competition and ensures a fair experience for all participants.

By participating in this online programming competition, you agree to abide by the principles of fair play and adhere to the rules and guidelines set forth. We reserve the right to investigate any suspected violations and take appropriate actions, including disqualification, if necessary

8. Disclaimers

We want to ensure that participants in this online programming competition are aware of certain disclaimers and limitations associated with the event. Please carefully read the following disclaimers before participating:

Technical Issues: While we strive to provide a smooth and uninterrupted experience, it is important to acknowledge that technical issues may arise during the competition. These issues could include, but are not limited to, server downtime, network disruptions, or software bugs. We reserve the right to address and resolve these technical issues promptly to ensure fair competition. In such cases, we may need to pause or temporarily suspend the competition to rectify the problems.
System Analysis and Bot Pairing: The competition system will analyze the performance of the submitted bots by pairing them against each other and allowing them to play a game. The outcome of these pairings will determine the evaluation and ranking of the bots. However, please note that the system's analysis and pairing algorithms are subject to limitations and may not always produce perfect or desired results. We will make reasonable efforts to ensure fairness and accuracy in the evaluation process, but we cannot guarantee flawless outcomes.
No Liability for Problems or Damage: While we take precautions to provide a secure and reliable competition environment, we cannot be held responsible for any problems, damages, or losses that may occur during the event. This includes, but is not limited to, data loss, system malfunctions, or any other issues that may arise from participating in the competition. Participants are responsible for ensuring the safety and backup of their own data and systems.
Code and Bot Security: Participants are responsible for the security and integrity of their own code and bots. We recommend taking necessary precautions to protect your intellectual property and prevent unauthorized access or tampering. We will not be held liable for any unauthorized use, modification, or disclosure of participants' code or bots.
Fair Play and Compliance: Participants are expected to adhere to the rules and guidelines outlined in the competition rulebook. Any violation of these rules, including cheating, exploiting vulnerabilities, or engaging in unethical practices, may result in disqualification from the competition. We reserve the right to take appropriate actions to maintain fair play and integrity throughout the event.

By participating in this online programming competition, you acknowledge and agree to these disclaimers. If you have any concerns or questions regarding these disclaimers, please reach out to us for clarification.

9. Amendments

In order to ensure fairness and adaptability throughout the competition, it may be necessary to make amendments or updates to the rules. These amendments will be made with the intention of improving the overall experience for participants and maintaining a level playing field.

We reserve the right to amend the rules of the competition at any time, and it is the responsibility of participants to stay updated with any changes. Amendments may include, but are not limited to, adjustments in evaluation criteria, submission guidelines, or any other aspect deemed necessary for the smooth operation of the competition.

Any amendments to the rules will be communicated to all participants through official channels, such as email notifications or announcements on the competition platform. It is important for participants to regularly check these channels to stay informed about any updates.

Upon receiving notification of an amendment, participants are expected to comply with the revised rules. Failure to adhere to the updated rules may result in disqualification or other appropriate actions as determined by the competition organizers.

We understand that amendments may introduce new challenges or considerations for participants. However, we assure you that any changes made will be done in a fair and transparent manner, with the best interests of all participants in mind.

If you have any questions or concerns regarding amendments to the rules, please do not hesitate to contact us. We are committed to providing clear and timely information to ensure a positive and engaging experience for all participants.

10. Contact Information

If participants have any questions, concerns, or need to contact us regarding the online programming competition, we are here to assist. Please find the contact information on the contact page.

We strive to provide prompt and helpful support to all participants. Whether you have inquiries about the competition rules, technical issues, or any other related matters, please don't hesitate to reach out to us.

Copyright © BotBattle FV 2023-2026
