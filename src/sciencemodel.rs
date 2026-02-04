use std::{fs::File, path::Path};

use bevy_ecs::{
    bundle::Bundle,
    component::Component,
    entity::Entity,
    hierarchy::ChildOf,
    resource::Resource,
    schedule::{IntoScheduleConfigs, ScheduleLabel, Schedules},
    system::{Commands, Query, Res, ResMut},
    world::World,
};
use bevy_math::UVec2;
use polars::prelude::*;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
    rngs::StdRng,
};

// ================================================================
// =============================== COMPONENTS======================
// ================================================================

#[derive(Component, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Age(u32);

impl Default for Age {
    fn default() -> Self {
        Self(23)
    }
}

#[derive(Component, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position(pub UVec2);

#[derive(Component, Default)]
pub struct MoveIntent {
    pub target: UVec2,
}

#[derive(Component, Default, Clone, Copy)]
struct Prestige {
    pub prestige: f64,
    pub prestige_visibility: f64,
    pub prestige_vanishing: f64,
}

#[derive(Component, Clone, Copy, Default)]
struct Curiosity {
    pub curiostity: f64,
    pub epsilon: f64,
}

#[derive(Component, Clone, Copy, Default, Debug)]
pub struct Visibility(pub f64);

#[derive(Component, Default, Clone, Copy)]
pub struct LastTileKnowledge(pub f64);

#[derive(Bundle, Default)]
struct Scientist {
    pub curiosity: Curiosity,
    pub prestige: Prestige,
    pub age: Age,
}

// ================================================================
// =============================== RESOURCES ======================
// ================================================================

#[derive(Resource)]
pub struct SimRng(pub StdRng);

#[derive(Resource, Clone, Copy)]
pub struct SpawnConfig {
    pub initial_curiosity: f64,
    pub epsilon: f64,
    pub min_age: u32,           // 23
    pub max_age_inclusive: u32, // 50
    pub spawn_rate: f64,
    pub new_agents_per_generation: u32,
    pub initial_agents: u32,
}

#[derive(Resource)]
struct PrestigeVanishingFactor(f64);

#[derive(Resource)]
pub struct GlobalAverages {
    pub avg_current_agent_knowledge: f64,
    pub agent_avg_distance: f64,
}

#[derive(Resource, Default)]
pub struct StepCounter(pub u32);

#[derive(Resource)]
pub struct DataFramesStore {
    pub model_df: DataFrame,
    pub agent_df: DataFrame,
    pub collect_agents: bool,
}

impl Default for DataFramesStore {
    fn default() -> Self {
        Self {
            model_df: DataFrame::new(vec![
                Column::new("step".into(), Vec::<u32>::new()),
                Column::new("mean_age".into(), Vec::<f64>::new()),
                Column::new("mean_prestige".into(), Vec::<f64>::new()),
                Column::new("mean_vanishing_prestige".into(), Vec::<f64>::new()),
                Column::new("mean_visibility_prestige".into(), Vec::<f64>::new()),
                Column::new("top1pct_prestige_share".into(), Vec::<f64>::new()),
                Column::new("top10pct_prestige_share".into(), Vec::<f64>::new()),
                Column::new("avg_current_agent_knowledge".into(), Vec::<f64>::new()),
                Column::new("explored_percentage".into(), Vec::<f64>::new()),
                Column::new(
                    "explored_weighted_by_initial_knowledge".into(),
                    Vec::<f64>::new(),
                ),
                Column::new("total_initial_knowledge".into(), Vec::<f64>::new()),
                Column::new("avg_knowledge_on_grid".into(), Vec::<f64>::new()),
                Column::new("best_knowledge".into(), Vec::<f64>::new()),
                Column::new("avg_distance_between_agents".into(), Vec::<f64>::new()),
                Column::new("percentage_knowledge_harvested".into(), Vec::<f64>::new()),
                Column::new("corr_prestige_local_merit".into(), Vec::<f64>::new()),
            ])
            .expect("model df"),
            agent_df: DataFrame::new(vec![
                Column::new("step".into(), Vec::<u32>::new()),
                Column::new("entity".into(), Vec::<u64>::new()), // store Entity as u64
                Column::new("prestige".into(), Vec::<f64>::new()),
                Column::new("prestige_vanishing".into(), Vec::<f64>::new()),
                Column::new("prestige_visibility".into(), Vec::<f64>::new()),
                Column::new("last_tile_knowledge".into(), Vec::<f64>::new()),
                Column::new("curiosity".into(), Vec::<f64>::new()),
                Column::new("local_merit".into(), Vec::<f64>::new()),
            ])
            .expect("agent df"),
            collect_agents: false,
        }
    }
}

#[derive(Resource)]
pub struct HarvestLevel(pub f64);

#[derive(Resource, Clone, Copy, PartialEq, PartialOrd, Ord, Eq)]
pub struct RetirementAge(pub u32);

#[derive(Resource)]
pub struct KnowledgeGrid {
    pub size: u32,
    pub cells: Vec<CellKnowledge>,
}

#[derive(Clone, Copy, Default)]
pub struct CellKnowledge {
    pub seen: bool,
    pub value: f64,
}

impl KnowledgeGrid {
    pub fn new(size: u32) -> Self {
        let n = (size as usize) * (size as usize);
        Self {
            size,
            cells: vec![CellKnowledge::default(); n],
        }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.size + x) as usize
    }

    /// 4-neighborhood on a torus (wraps).
    fn neighbors(&self, pos: UVec2) -> Vec<UVec2> {
        let s = self.size;
        debug_assert!(pos.x < s && pos.y < s);

        let x = pos.x;
        let y = pos.y;

        let left = (x + s - 1) % s;
        let right = (x + 1) % s;
        let down = (y + s - 1) % s;
        let up = (y + 1) % s;

        vec![
            UVec2::new(left, y),
            UVec2::new(right, y),
            UVec2::new(x, down),
            UVec2::new(x, up),
        ]
    }
    pub fn get(&self, x: u32, y: u32) -> Option<CellKnowledge> {
        if x < self.size && y < self.size {
            Some(self.cells[self.idx(x, y)])
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, x: u32, y: u32) -> Option<&mut CellKnowledge> {
        if x < self.size && y < self.size {
            let i = self.idx(x, y);
            Some(&mut self.cells[i])
        } else {
            None
        }
    }
}

/// Circular (torus) L1 distance in 1D: min(|a-b|, size-|a-b|)
#[inline]
fn torus_1d(a: u32, b: u32, size: u32) -> u32 {
    let d = a.abs_diff(b);
    d.min(size - d)
}

/// Torus (wrap-around) Manhattan distance
#[inline]
fn torus_manhattan(a: UVec2, b: UVec2, size: u32) -> u32 {
    torus_1d(a.x, b.x, size) + torus_1d(a.y, b.y, size)
}

fn compute_visibility(
    me: Entity,
    candidate_pos: UVec2,
    globals: &GlobalAverages,
    grid_size: u32,
    others: &[(Entity, Position, f64)],
) -> f64 {
    let mut wsum = 0.0;
    let mut wtot = 0.0;

    let mut sum = 0.0;
    let mut cnt = 0.0;

    for (e, op, w) in others.iter() {
        if *e == me {
            continue;
        }

        let d = torus_manhattan(candidate_pos, op.0, grid_size) as f64;

        sum += d;
        cnt += 1.0;

        if *w != 0.0 {
            wsum += d * (*w);
            wtot += *w;
        }
    }

    let avg_agent_distance_unweighted = if cnt > 0.0 { sum / cnt } else { 0.0 };

    let avg_agent_distance_p = if wtot > 0.0 {
        wsum / wtot
    } else {
        avg_agent_distance_unweighted
    };

    (globals.agent_avg_distance + 1.0) / (avg_agent_distance_p + 1.0)
}

fn compute_all_rewards(
    visibility_p: f64,
    curiosity: f64,
    novelty: f64,
    globals: &GlobalAverages,
) -> f64 {
    let knowledge_term = novelty / globals.avg_current_agent_knowledge;

    knowledge_term.powf(curiosity) * visibility_p.powf(1.0 - curiosity)
}

fn compute_visibilities(
    globals: Res<GlobalAverages>,
    grid: Res<KnowledgeGrid>,
    mut q_agents: Query<(Entity, &Position, &mut Visibility)>,
    q_positions: Query<(Entity, &Position, &Prestige)>,
) {
    let others: Vec<(Entity, Position, f64)> = q_positions
        .iter()
        .map(|(e, p, prest)| (e, *p, prest.prestige_vanishing))
        .collect();

    for (me, pos, mut visibility) in q_agents.iter_mut() {
        visibility.0 = compute_visibility(me, pos.0, &globals, grid.size, &others);
    }
}

fn decide_moves(
    globals: Res<GlobalAverages>,
    grid: Res<KnowledgeGrid>,
    mut q_agents: Query<(Entity, &Position, &Curiosity, &Prestige, &Visibility)>,
    q_positions: Query<(Entity, &Position, &Prestige)>,
    mut rand_gen: ResMut<SimRng>,
    mut commands: Commands,
) {
    let others: Vec<(Entity, Position, f64)> = q_positions
        .iter()
        .map(|(e, p, prest)| (e, *p, prest.prestige_vanishing))
        .collect();

    for (me, pos, curiosity, _my_prest, vis) in q_agents.iter_mut() {
        let novelty = grid.get(pos.0.x, pos.0.y).unwrap().value;
        let current_reward = compute_all_rewards(vis.0, curiosity.curiostity, novelty, &globals);

        let mut best_reward = current_reward;
        let mut target = pos.0;

        for npos in grid.neighbors(pos.0) {
            let neighbor_novelty = grid.get(npos.x, npos.y).unwrap().value;

            let noise: f64 = rand_gen.0.random();
            let novelty_noisy = neighbor_novelty * (1.0 + curiosity.epsilon * noise);

            let vis2 = compute_visibility(me, npos, &globals, grid.size, &others);
            let reward = compute_all_rewards(vis2, curiosity.curiostity, novelty_noisy, &globals);

            if reward > best_reward {
                best_reward = reward;
                target = npos;
            }
        }

        commands
            .entity(me)
            .insert_if(MoveIntent { target }, || best_reward - current_reward > 0.0);
    }
}

fn apply_moves(mut q: Query<(Entity, &mut Position, &MoveIntent)>, mut commands: Commands) {
    for (scientist, mut pos, intent) in q.iter_mut() {
        println!("Agent moved from {:?} to {:?}", pos, intent.target);
        pos.0 = intent.target;
        commands.entity(scientist).remove::<MoveIntent>();
    }
}

fn farm_knowledge(
    q_positions: Query<&Position>,
    mut grid: ResMut<KnowledgeGrid>,
    harvest: Res<HarvestLevel>,
) {
    for pos in q_positions {
        let knowledge_cell = grid.get_mut(pos.0.x, pos.0.y).unwrap();
        knowledge_cell.value *= 1.0 - harvest.0;
        knowledge_cell.seen = true;
    }
}

fn increase_prestige(
    grid: Res<KnowledgeGrid>,
    presite_vanishing_factor: Res<PrestigeVanishingFactor>,
    q_prestige: Query<(&Position, &mut Prestige)>,
) {
    for (position, mut presige) in q_prestige {
        let knowledge = grid.get(position.0.x, position.0.y).unwrap().value;
        presige.prestige += knowledge;
        let new_vanishing_factor =
            presige.prestige_vanishing * presite_vanishing_factor.0 + knowledge;
        presige.prestige_vanishing = new_vanishing_factor;
    }
}

fn spawn_agent(
    commands: &mut Commands,
    rng: &mut SimRng,
    grid: &KnowledgeGrid,
    spawn: &SpawnConfig,
) -> Entity {
    let x = rng.0.random_range(0..grid.size);
    let y = rng.0.random_range(0..grid.size);
    let pos = UVec2::new(x, y);

    let age: u32 = rng.0.random_range(spawn.min_age..=spawn.max_age_inclusive);

    let last_tile_knowledge = grid.get(pos.x, pos.y).map(|c| c.value).unwrap_or(0.0);

    commands
        .spawn((
            Scientist {
                curiosity: Curiosity {
                    curiostity: spawn.initial_curiosity,
                    epsilon: spawn.epsilon,
                },
                prestige: Prestige::default(),
                age: Age(age),
            },
            Position(pos),
            Visibility(0.0),
            LastTileKnowledge(last_tile_knowledge),
        ))
        .id()
}

fn spawn_agent_with_supervisor(
    commands: &mut Commands,
    rng: &mut SimRng,
    grid: &KnowledgeGrid,
    spawn: &SpawnConfig,
    supervisor: (Entity, UVec2),
) -> Entity {
    let x = supervisor.1.x;
    let y = supervisor.1.y;
    let pos = UVec2::new(x, y);

    let age: u32 = rng.0.random_range(spawn.min_age..=spawn.max_age_inclusive);

    let last_tile_knowledge = grid.get(pos.x, pos.y).map(|c| c.value).unwrap_or(0.0);

    commands
        .spawn((
            Scientist {
                curiosity: Curiosity {
                    curiostity: spawn.initial_curiosity,
                    epsilon: spawn.epsilon,
                },
                prestige: Prestige::default(),
                age: Age(age),
            },
            Position(pos),
            Visibility(0.0),
            LastTileKnowledge(last_tile_knowledge),
            ChildOf(supervisor.0),
        ))
        .id()
}

fn pick_supervisor_weighted_index(
    q: &Query<(Entity, &Prestige, &Position)>,
    rng: &mut SimRng,
) -> Option<Entity> {
    let mut entities: Vec<Entity> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    for (e, p, _) in q.iter() {
        let w = p.prestige;
        if w.is_finite() && w > 0.0 {
            entities.push(e);
            weights.push(w);
        }
    }

    if entities.is_empty() {
        return None;
    }

    let dist = WeightedIndex::new(&weights).ok()?;
    let idx = dist.sample(&mut rng.0);
    Some(entities[idx])
}

fn setup_resources(mut commands: Commands) {
    commands.insert_resource(KnowledgeGrid::new(20));
    commands.insert_resource(RetirementAge(63));

    commands.insert_resource(GlobalAverages {
        avg_current_agent_knowledge: 1.0,
        agent_avg_distance: 1.0,
    });
    commands.insert_resource(HarvestLevel(0.01));
    commands.insert_resource(PrestigeVanishingFactor(0.95));

    commands.insert_resource(SimRng(StdRng::seed_from_u64(0)));
    commands.insert_resource(SpawnConfig {
        initial_curiosity: 0.5,
        epsilon: 0.1,
        min_age: 23,
        max_age_inclusive: 50,
        spawn_rate: 0.8,
        new_agents_per_generation: 10,
        initial_agents: 10,
    });
    commands.init_resource::<DataFramesStore>();
    commands.init_resource::<StepCounter>();
}

fn spawn_initial_agents(
    mut commands: Commands,
    mut rng: ResMut<SimRng>,
    grid: Res<KnowledgeGrid>,
    spawn: Res<SpawnConfig>,
) {
    for _ in 0..spawn.initial_agents {
        spawn_agent(&mut commands, &mut rng, &grid, &spawn);
    }
}

fn increase_age(q_ages: Query<&mut Age>) {
    for mut age in q_ages {
        age.0 += 1;
    }
}

fn retire(
    q_ages: Query<(Entity, &Age)>,
    mut commands: Commands,
    retirement_age: Res<RetirementAge>,
    mut rng: ResMut<SimRng>,
) {
    for (e, age) in q_ages.iter().filter(|(_, age)| age.0 > retirement_age.0) {
        if rng.0.random_bool(age.0 as f64 / 100.0) {
            commands.entity(e).despawn()
        }
    }
}

fn update_step_counter(mut step_counter: ResMut<StepCounter>) {
    step_counter.0 += 1;
}

fn spawn_new_agents(
    spawn: Res<SpawnConfig>,
    mut commands: Commands,
    mut rng: ResMut<SimRng>,
    grid: Res<KnowledgeGrid>,
    q_supervisors: Query<(Entity, &Prestige, &Position)>,
) {
    for _ in 1..=spawn.new_agents_per_generation {
        if rng.0.random_bool(spawn.spawn_rate)
            && let Some(supervisor) = pick_supervisor_weighted_index(&q_supervisors, &mut rng)
        {
            let supervisor = q_supervisors
                .get(supervisor)
                .map(|(a, _, c)| (a, c.0))
                .unwrap();
            spawn_agent_with_supervisor(&mut commands, &mut rng, &grid, &spawn, supervisor);
        }
    }
}

fn update_averages(
    mut globals: ResMut<GlobalAverages>,
    q_lasttileknowledge: Query<&LastTileKnowledge>,
    q_pos: Query<(Entity, &Position)>,
    grid: Res<KnowledgeGrid>,
) {
    globals.avg_current_agent_knowledge = q_lasttileknowledge
        .iter()
        .fold(0.0, |acc: f64, el| acc + el.0)
        / q_lasttileknowledge.iter().len() as f64;

    let positions: Vec<(Entity, UVec2)> = q_pos.iter().map(|(e, p)| (e, p.0)).collect();

    if positions.len() < 2 {
        globals.agent_avg_distance = 0.0;
        return;
    }

    let mut sum_of_means = 0.0;
    let mut num_agents = 0.0;

    for (e, pa) in positions.iter() {
        let mut sum = 0.0;
        let mut n = 0.0;

        for (e2, pb) in positions.iter() {
            if e2 == e {
                continue;
            }
            sum += torus_manhattan(*pa, *pb, grid.size) as f64;
            n += 1.0;
        }

        if n > 0.0 {
            sum_of_means += sum / n;
            num_agents += 1.0;
        }
    }

    globals.agent_avg_distance = sum_of_means / num_agents;
}

fn collect_to_datastore(
    mut store: ResMut<DataFramesStore>,
    mut step: ResMut<StepCounter>,
    grid: Res<KnowledgeGrid>,
    globals: Res<GlobalAverages>,
    q_agents: Query<(
        Entity,
        &Age,
        &Prestige,
        &Visibility,
        &Curiosity,
        &LastTileKnowledge,
        &Position,
    )>,
) {
    let s = step.0;
    step.0 += 1;

    let n_agents = q_agents.iter().len().max(1) as f64;

    // --- model-level aggregates ---
    let mut sum_age = 0.0;
    let mut sum_prestige = 0.0;
    let mut sum_prestige_vanishing = 0.0;
    let mut sum_prestige_visibility = 0.0;

    for (_, age, p, _vis, _cur, _last, _pos) in q_agents.iter() {
        sum_age += age.0 as f64;
        sum_prestige += p.prestige;
        sum_prestige_vanishing += p.prestige_vanishing;
        sum_prestige_visibility += p.prestige_visibility;
    }

    let mean_age = sum_age / n_agents;
    let mean_prestige = sum_prestige / n_agents;
    let mean_vanishing_prestige = sum_prestige_vanishing / n_agents;
    let mean_visibility_prestige = sum_prestige_visibility / n_agents;

    let mut explored = 0.0;
    let mut explored_weighted = 0.0;
    let mut total_initial_knowledge = 0.0;

    for c in &grid.cells {
        if c.seen {
            explored += 1.0;
            explored_weighted += c.value;
        }
        total_initial_knowledge += c.value;
    }

    let n_cells = grid.cells.len().max(1) as f64;
    let explored_percentage = explored / n_cells;
    let explored_weighted_by_initial_knowledge = if total_initial_knowledge != 0.0 {
        explored_weighted / total_initial_knowledge
    } else {
        0.0
    };

    let mut sum_grid = 0.0;
    let mut best_knowledge = f64::NEG_INFINITY;
    for c in &grid.cells {
        sum_grid += c.value;
        best_knowledge = best_knowledge.max(c.value);
    }
    let avg_knowledge_on_grid = sum_grid / n_cells;

    let top1pct_prestige_share = 0.0;
    let top10pct_prestige_share = 0.0;
    let percentage_knowledge_harvested = 0.0;
    let corr_prestige_local_merit = 0.0;

    let model_row = df![
        "step" => [s],
        "mean_age" => [mean_age],
        "mean_prestige" => [mean_prestige],
        "mean_vanishing_prestige" => [mean_vanishing_prestige],
        "mean_visibility_prestige" => [mean_visibility_prestige],
        "top1pct_prestige_share" => [top1pct_prestige_share],
        "top10pct_prestige_share" => [top10pct_prestige_share],
        "avg_current_agent_knowledge" => [globals.avg_current_agent_knowledge],
        "explored_percentage" => [explored_percentage],
        "explored_weighted_by_initial_knowledge" => [explored_weighted_by_initial_knowledge],
        "total_initial_knowledge" => [total_initial_knowledge],
        "avg_knowledge_on_grid" => [avg_knowledge_on_grid],
        "best_knowledge" => [best_knowledge],
        "avg_distance_between_agents" => [globals.agent_avg_distance],
        "percentage_knowledge_harvested" => [percentage_knowledge_harvested],
        "corr_prestige_local_merit" => [corr_prestige_local_merit]
    ]
    .expect("model row df");

    store
        .model_df
        .vstack_mut(&model_row)
        .expect("append model row");

    // --- agent-level rows (optional) ---
    if store.collect_agents {
        // one row per agent
        for (e, _age, p, _vis, c, last, pos) in q_agents.iter() {
            let local_merit = grid.get(pos.0.x, pos.0.y).map(|ck| ck.value).unwrap_or(0.0);

            let agent_row = df![
                "step" => [s],
                "entity" => [e.to_bits()],
                "prestige" => [p.prestige],
                "prestige_vanishing" => [p.prestige_vanishing],
                "prestige_visibility" => [p.prestige_visibility],
                "last_tile_knowledge" => [last.0],
                "curiosity" => [c.curiostity],
                "local_merit" => [local_merit]
            ]
            .expect("agent row df");

            store
                .agent_df
                .vstack_mut(&agent_row)
                .expect("append agent row");
        }
    }
}

fn save_datastore_to_csv(store: Res<DataFramesStore>) {
    let out_dir = Path::new("output");
    let _ = std::fs::create_dir_all(out_dir);

    let model_path = out_dir.join("model.csv");
    let agent_path = out_dir.join("agents.csv");

    let mut f_model = File::create(model_path).expect("create model.csv");
    CsvWriter::new(&mut f_model)
        .include_header(true)
        .finish(&mut store.model_df.clone())
        .expect("write model.csv");

    let mut f_agent = File::create(agent_path).expect("create agents.csv");
    CsvWriter::new(&mut f_agent)
        .include_header(true)
        .finish(&mut store.agent_df.clone())
        .expect("write agents.csv");
}

// ================================================================
// =============================== SCHEDULES ======================
// ================================================================
#[derive(ScheduleLabel, Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct StartUp;

#[derive(ScheduleLabel, Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct Update;

fn main() {
    let mut world = World::new();
    world
        .get_resource_or_init::<Schedules>()
        .add_systems(StartUp, (setup_resources, spawn_initial_agents).chain())
        .add_systems(
            Update,
            (
                (
                    compute_visibilities,
                    decide_moves,
                    apply_moves,
                    farm_knowledge,
                    increase_prestige,
                    update_averages.after(spawn_new_agents),
                )
                    .chain(),
                (increase_age, retire).chain(),
                spawn_new_agents,
                update_step_counter,
                collect_to_datastore.after(update_averages),
                save_datastore_to_csv.after(collect_to_datastore),
            ),
        );
    world.run_schedule(StartUp);
    for _ in 1..400 {
        world.run_schedule(Update);
    }
}
