#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PLAYER_COUNT 6
#define BOARD_RADIUS 5
#define BOARD_TILE_COUNT 91
#define BOARD_DIAMETER 11
#define BOARD_OFFSET 5
#define MOVE_INTERVAL 10000
#define NO_GEM_TICKS_THRESHOLD 50000
#define MAX_BODY_LEN 91

typedef struct {
    int alive;
    int facing;
    int pending_turn;
    int growth_pending;
    int length;
    int body_q[MAX_BODY_LEN];
    int body_r[MAX_BODY_LEN];
} Player;

typedef struct {
    int no_gem_ticks;
    Player players[PLAYER_COUNT];
    unsigned char gems[BOARD_TILE_COUNT];
} CoreState;

static CoreState G;
static int TILE_INDEX_MAP[BOARD_DIAMETER][BOARD_DIAMETER];
static int TILE_Q[BOARD_TILE_COUNT];
static int TILE_R[BOARD_TILE_COUNT];
static int TILE_MAP_READY = 0;

static const int CORNER_SPAWNS[PLAYER_COUNT][2] = {
    {0, -5},
    {5, -5},
    {5, 0},
    {0, 5},
    {-5, 5},
    {-5, 0},
};

static const int CORNER_FACINGS[PLAYER_COUNT] = {3, 4, 5, 0, 1, 2};

static const int INITIAL_GEMS[PLAYER_COUNT][2] = {
    {0, -2},
    {2, -2},
    {2, 0},
    {0, 2},
    {-2, 2},
    {-2, 0},
};

static const int DIRECTIONS[6][2] = {
    {0, -1},
    {1, -1},
    {1, 0},
    {0, 1},
    {-1, 1},
    {-1, 0},
};

static int abs_i(int x) {
    return x < 0 ? -x : x;
}

static int wrap_facing(int f) {
    int out = f % 6;
    if (out < 0) {
        out += 6;
    }
    return out;
}

static int effective_facing(int pid) {
    return wrap_facing(G.players[pid].facing + G.players[pid].pending_turn);
}

static int is_inside(int q, int r) {
    int s = -q - r;
    int a = abs_i(q);
    int b = abs_i(r);
    int c = abs_i(s);
    int m = a > b ? a : b;
    m = m > c ? m : c;
    return m <= BOARD_RADIUS;
}

static void init_tile_map(void) {
    if (TILE_MAP_READY) {
        return;
    }

    for (int qi = 0; qi < BOARD_DIAMETER; qi++) {
        for (int ri = 0; ri < BOARD_DIAMETER; ri++) {
            TILE_INDEX_MAP[qi][ri] = -1;
        }
    }

    int idx = 0;
    for (int q = -BOARD_RADIUS; q <= BOARD_RADIUS; q++) {
        for (int r = -BOARD_RADIUS; r <= BOARD_RADIUS; r++) {
            if (!is_inside(q, r)) {
                continue;
            }
            TILE_INDEX_MAP[q + BOARD_OFFSET][r + BOARD_OFFSET] = idx;
            TILE_Q[idx] = q;
            TILE_R[idx] = r;
            idx++;
        }
    }

    TILE_MAP_READY = 1;
}

static int tile_index(int q, int r) {
    if (q < -BOARD_RADIUS || q > BOARD_RADIUS || r < -BOARD_RADIUS || r > BOARD_RADIUS) {
        return -1;
    }
    return TILE_INDEX_MAP[q + BOARD_OFFSET][r + BOARD_OFFSET];
}

static int occupied_by_player(int q, int r) {
    for (int pid = 0; pid < PLAYER_COUNT; pid++) {
        Player *p = &G.players[pid];
        for (int i = 0; i < p->length; i++) {
            if (p->body_q[i] == q && p->body_r[i] == r) {
                return pid;
            }
        }
    }
    return -1;
}

static void rotate_axial(int q, int r, int clockwise_steps, int *out_q, int *out_r) {
    int x = q;
    int z = r;
    int y = -x - z;

    int steps = ((clockwise_steps % 6) + 6) % 6;
    for (int i = 0; i < steps; i++) {
        int nx = -z;
        int ny = -x;
        int nz = -y;
        x = nx;
        y = ny;
        z = nz;
    }

    *out_q = x;
    *out_r = z;
}

void tc_init(void) {
    init_tile_map();
    memset(&G, 0, sizeof(G));

    for (int i = 0; i < PLAYER_COUNT; i++) {
        Player *p = &G.players[i];
        p->alive = 1;
        p->facing = CORNER_FACINGS[i];
        p->pending_turn = 0;
        p->growth_pending = 0;
        p->length = 1;
        p->body_q[0] = CORNER_SPAWNS[i][0];
        p->body_r[0] = CORNER_SPAWNS[i][1];
    }

    for (int i = 0; i < PLAYER_COUNT; i++) {
        int q = INITIAL_GEMS[i][0];
        int r = INITIAL_GEMS[i][1];
        int idx = tile_index(q, r);
        if (idx >= 0) {
            G.gems[idx] = 1;
        }
    }

    G.no_gem_ticks = 0;
}

void tc_mark_dead(int pid) {
    if (pid < 0 || pid >= PLAYER_COUNT) {
        return;
    }
    G.players[pid].alive = 0;
}

void tc_set_turn(int pid, int rel_turn) {
    if (pid < 0 || pid >= PLAYER_COUNT) {
        return;
    }
    if (rel_turn < -1 || rel_turn > 1) {
        return;
    }
    G.players[pid].pending_turn = rel_turn;
}

void tc_set_turn_raw(int pid, int value) {
    if (pid < 0 || pid >= PLAYER_COUNT) {
        return;
    }
    if (value < -1 || value > 1) {
        return;
    }
    G.players[pid].pending_turn = value;
}

void tc_config_player(int pid, int alive, int q, int r, int facing) {
    if (pid < 0 || pid >= PLAYER_COUNT) {
        return;
    }
    if (!is_inside(q, r)) {
        return;
    }

    Player *p = &G.players[pid];
    p->alive = alive ? 1 : 0;
    p->facing = wrap_facing(facing);
    p->pending_turn = 0;
    p->growth_pending = 0;
    p->length = 1;
    p->body_q[0] = q;
    p->body_r[0] = r;
}

int tc_get_turn(int pid) {
    if (pid < 0 || pid >= PLAYER_COUNT) {
        return 0;
    }
    return G.players[pid].pending_turn;
}

void tc_rel_to_abs(int pid, int rq, int rr, int *out_q, int *out_r) {
    if (out_q == NULL || out_r == NULL) {
        return;
    }

    if (pid < 0 || pid >= PLAYER_COUNT) {
        *out_q = 0;
        *out_r = 0;
        return;
    }

    int rot_q = 0;
    int rot_r = 0;
    rotate_axial(rq, rr, effective_facing(pid), &rot_q, &rot_r);

    Player *p = &G.players[pid];
    *out_q = p->body_q[0] + rot_q;
    *out_r = p->body_r[0] + rot_r;
}

void tc_get_tile_abs(int q, int r, int *exists, int *is_empty, int *player_id, int *has_gem) {
    if (exists == NULL || is_empty == NULL || player_id == NULL || has_gem == NULL) {
        return;
    }

    int idx = tile_index(q, r);
    if (idx < 0) {
        *exists = 0;
        *is_empty = 0;
        *player_id = -1;
        *has_gem = 0;
        return;
    }

    int pid = occupied_by_player(q, r);
    if (pid >= 0) {
        *exists = 1;
        *is_empty = 0;
        *player_id = pid;
        *has_gem = 0;
        return;
    }

    int gem = G.gems[idx];

    *exists = 1;
    *is_empty = gem ? 0 : 1;
    *player_id = -1;
    *has_gem = gem ? 1 : 0;
}

void tc_get_tile_rel(int pid, int rq, int rr, int *exists, int *is_empty, int *player_id, int *has_gem) {
    int q = 0;
    int r = 0;
    tc_rel_to_abs(pid, rq, rr, &q, &r);
    tc_get_tile_abs(q, r, exists, is_empty, player_id, has_gem);
}

void tc_get_player_info(int pid, int *alive, int *head_q, int *head_r, int *head_facing, int *length) {
    if (alive == NULL || head_q == NULL || head_r == NULL || head_facing == NULL || length == NULL) {
        return;
    }

    if (pid < 0 || pid >= PLAYER_COUNT) {
        *alive = 0;
        *head_q = 0;
        *head_r = 0;
        *head_facing = 0;
        *length = 0;
        return;
    }

    Player *p = &G.players[pid];
    *alive = p->alive ? 1 : 0;
    *head_q = p->body_q[0];
    *head_r = p->body_r[0];
    *head_facing = effective_facing(pid);
    *length = p->length;
}

static void prepend_to_body(Player *p, int q, int r, int keep_length) {
    int old_len = p->length;
    int move_len = old_len;
    if (move_len > MAX_BODY_LEN - 1) {
        move_len = MAX_BODY_LEN - 1;
    }

    for (int i = move_len - 1; i >= 0; i--) {
        p->body_q[i + 1] = p->body_q[i];
        p->body_r[i + 1] = p->body_r[i];
    }

    p->body_q[0] = q;
    p->body_r[0] = r;

    if (keep_length) {
        int new_len = old_len + 1;
        if (new_len > MAX_BODY_LEN) {
            new_len = MAX_BODY_LEN;
        }
        p->length = new_len;
    } else {
        p->length = old_len;
    }
}

void tc_resolve_step(int *death_reasons6, int *gem_picked6, int *spawn_from_pickups, int *spawn_from_starvation) {
    int desired_face[PLAYER_COUNT];
    int desired_q[PLAYER_COUNT];
    int desired_r[PLAYER_COUNT];

    if (spawn_from_pickups != NULL) {
        *spawn_from_pickups = 0;
    }
    if (spawn_from_starvation != NULL) {
        *spawn_from_starvation = 0;
    }

    for (int i = 0; i < PLAYER_COUNT; i++) {
        if (death_reasons6 != NULL) {
            death_reasons6[i] = 0;
        }
        if (gem_picked6 != NULL) {
            gem_picked6[i] = 0;
        }
        desired_face[i] = 0;
        desired_q[i] = 0;
        desired_r[i] = 0;
    }

    // Compute intended destinations.
    for (int pid = 0; pid < PLAYER_COUNT; pid++) {
        Player *p = &G.players[pid];
        if (!p->alive) {
            continue;
        }

        int face = wrap_facing(p->facing + p->pending_turn);
        int dq = DIRECTIONS[face][0];
        int dr = DIRECTIONS[face][1];
        int q = p->body_q[0] + dq;
        int r = p->body_r[0] + dr;

        desired_face[pid] = face;
        desired_q[pid] = q;
        desired_r[pid] = r;
    }

    // Wall / occupied deaths.
    for (int pid = 0; pid < PLAYER_COUNT; pid++) {
        Player *p = &G.players[pid];
        if (!p->alive) {
            continue;
        }

        int q = desired_q[pid];
        int r = desired_r[pid];

        if (!is_inside(q, r)) {
            if (death_reasons6 != NULL) {
                death_reasons6[pid] = 1;
            }
            continue;
        }

        int occ = occupied_by_player(q, r);
        if (occ >= 0) {
            if (death_reasons6 != NULL) {
                death_reasons6[pid] = 2;
            }
        }
    }

    // Contested destination deaths.
    for (int a = 0; a < PLAYER_COUNT; a++) {
        Player *pa = &G.players[a];
        if (!pa->alive) {
            continue;
        }
        if (death_reasons6 != NULL && death_reasons6[a] != 0) {
            continue;
        }

        for (int b = a + 1; b < PLAYER_COUNT; b++) {
            Player *pb = &G.players[b];
            if (!pb->alive) {
                continue;
            }
            if (death_reasons6 != NULL && death_reasons6[b] != 0) {
                continue;
            }

            if (desired_q[a] == desired_q[b] && desired_r[a] == desired_r[b]) {
                if (death_reasons6 != NULL) {
                    death_reasons6[a] = 3;
                    death_reasons6[b] = 3;
                }
            }
        }
    }

    // Apply deaths.
    for (int pid = 0; pid < PLAYER_COUNT; pid++) {
        if (death_reasons6 != NULL && death_reasons6[pid] != 0) {
            G.players[pid].alive = 0;
        }
    }

    int consumed_count = 0;

    // Move survivors.
    for (int pid = 0; pid < PLAYER_COUNT; pid++) {
        Player *p = &G.players[pid];
        if (!p->alive) {
            continue;
        }
        if (death_reasons6 != NULL && death_reasons6[pid] != 0) {
            continue;
        }

        int q = desired_q[pid];
        int r = desired_r[pid];

        p->facing = desired_face[pid];
        p->pending_turn = 0;

        int idx = tile_index(q, r);
        int consumed_gem = (idx >= 0 && G.gems[idx]) ? 1 : 0;
        if (consumed_gem) {
            G.gems[idx] = 0;
            p->growth_pending += 1;
            consumed_count += 1;
            if (gem_picked6 != NULL) {
                gem_picked6[pid] = 1;
            }
        }

        int keep_length = 0;
        if (p->growth_pending > 0) {
            p->growth_pending -= 1;
            keep_length = 1;
        }

        prepend_to_body(p, q, r, keep_length);
    }

    if (spawn_from_pickups != NULL) {
        *spawn_from_pickups = consumed_count;
    }

    if (consumed_count > 0) {
        G.no_gem_ticks = 0;
    } else {
        G.no_gem_ticks += MOVE_INTERVAL;
        if (G.no_gem_ticks >= NO_GEM_TICKS_THRESHOLD) {
            if (spawn_from_starvation != NULL) {
                *spawn_from_starvation = 1;
            }
        }
    }
}

int tc_list_spawn_candidates(int *qs, int *rs, int max_out) {
    int count = 0;

    for (int idx = 0; idx < BOARD_TILE_COUNT; idx++) {
        int q = TILE_Q[idx];
        int r = TILE_R[idx];

        if (G.gems[idx]) {
            continue;
        }

        if (occupied_by_player(q, r) >= 0) {
            continue;
        }

        if (qs != NULL && rs != NULL && count < max_out) {
            qs[count] = q;
            rs[count] = r;
        }
        count++;
    }

    return count;
}

int tc_add_gem(int q, int r) {
    int idx = tile_index(q, r);
    if (idx < 0) {
        return 0;
    }

    if (G.gems[idx]) {
        return 0;
    }

    if (occupied_by_player(q, r) >= 0) {
        return 0;
    }

    G.gems[idx] = 1;
    return 1;
}

void tc_get_player_state(
    int pid,
    int *alive,
    int *facing,
    int *pending,
    int *length,
    int *body_q,
    int *body_r,
    int max_len
) {
    if (alive == NULL || facing == NULL || pending == NULL || length == NULL) {
        return;
    }

    if (pid < 0 || pid >= PLAYER_COUNT) {
        *alive = 0;
        *facing = 0;
        *pending = 0;
        *length = 0;
        return;
    }

    Player *p = &G.players[pid];
    *alive = p->alive ? 1 : 0;
    *facing = p->facing;
    *pending = p->pending_turn;
    *length = p->length;

    if (body_q != NULL && body_r != NULL && max_len > 0) {
        int n = p->length;
        if (n > max_len) {
            n = max_len;
        }
        for (int i = 0; i < n; i++) {
            body_q[i] = p->body_q[i];
            body_r[i] = p->body_r[i];
        }
    }
}

int tc_get_gems(int *qs, int *rs, int max_out) {
    int count = 0;

    for (int idx = 0; idx < BOARD_TILE_COUNT; idx++) {
        if (G.gems[idx]) {
            if (qs != NULL && rs != NULL && count < max_out) {
                qs[count] = TILE_Q[idx];
                rs[count] = TILE_R[idx];
            }
            count++;
        }
    }

    return count;
}
