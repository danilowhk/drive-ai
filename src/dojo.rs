use crate::car::Car;
use crate::car::CarBundle;
use crate::configs;
use crate::enemy::Enemy;
use bevy::ecs::system::SystemState;
use bevy::log;
use bevy::prelude::*;
use bevy_tokio_tasks::TaskContext;
use bevy_tokio_tasks::{TokioTasksPlugin, TokioTasksRuntime};
use bigdecimal::ToPrimitive;
use dojo_client::contract::world::WorldContract;
use starknet::accounts::SingleOwnerAccount;
use starknet::core::types::{BlockId, BlockTag, FieldElement};
use starknet::core::utils::cairo_short_string_to_felt;
use starknet::providers::jsonrpc::HttpTransport;
use starknet::providers::JsonRpcClient;
use starknet::signers::{LocalWallet, SigningKey};
use std::str::FromStr;
use tokio::sync::mpsc;
use url::Url;

pub struct DojoPlugin;

impl Plugin for DojoPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(TokioTasksPlugin::default())
            .add_startup_systems((
                setup,
                spawn_racers_thread,
                drive_thread,
                update_vehicle_thread,
                update_enemies_thread,
            ))
            .add_system(sync_dojo_state);
    }
}

fn setup(mut commands: Commands) {
    commands.spawn(DojoSyncTime::from_seconds(configs::DOJO_SYNC_INTERVAL));
}

#[derive(Component)]
struct DojoSyncTime {
    timer: Timer,
}

impl DojoSyncTime {
    fn from_seconds(duration: f32) -> Self {
        Self {
            timer: Timer::from_seconds(duration, TimerMode::Repeating),
        }
    }
}

fn sync_dojo_state(
    mut dojo_sync_time: Query<&mut DojoSyncTime>,
    time: Res<Time>,
    drive: Res<DriveCommand>,
    update_vehicle: Res<UpdateVehicleCommand>,
    update_enemies: Res<UpdateEnemiesCommand>,
) {
    let mut dojo_time = dojo_sync_time.single_mut();

    if dojo_time.timer.just_finished() {
        dojo_time.timer.reset();

        if let Err(e) = update_vehicle.try_send() {
            log::error!("{e}");
        }
        if let Err(e) = drive.try_send() {
            log::error!("{e}");
        }
        if let Err(e) = update_enemies.try_send() {
            log::error!("{e}");
        }
    } else {
        dojo_time.timer.tick(time.delta());
    }
}

fn spawn_racers_thread(runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(SpawnRacersCommand(tx));

    runtime.spawn_background_task(|mut ctx| async move {
        let url = Url::parse(configs::JSON_RPC_ENDPOINT).unwrap();
        let account_address = FieldElement::from_str(configs::ACCOUNT_ADDRESS).unwrap();
        let account = SingleOwnerAccount::new(
            JsonRpcClient::new(HttpTransport::new(url)),
            LocalWallet::from_signing_key(SigningKey::from_secret_scalar(
                FieldElement::from_str(configs::ACCOUNT_SECRET_KEY).unwrap(),
            )),
            account_address,
            cairo_short_string_to_felt("KATANA").unwrap(),
        );

        let world_address = FieldElement::from_str(configs::WORLD_ADDRESS).unwrap();
        let block_id = BlockId::Tag(BlockTag::Latest);

        let world = WorldContract::new(world_address, &account);

        let spawn_racer_system = world.system("spawn_racer", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let mut dojo_ids = vec![];
            for i in 0..configs::NUM_AI_CARS {
                let dojo_id = FieldElement::from_dec_str(&i.to_string()).unwrap();

                dojo_ids.push(dojo_id);

                match spawn_racer_system.execute(vec![dojo_id]).await {
                    Ok(_) => {
                        log::info!("Run drive system");

                        ctx.run_on_main_thread(move |ctx| {
                            let asset_server = ctx.world.get_resource::<AssetServer>().unwrap();
                            ctx.world.spawn(CarBundle::new(&asset_server, dojo_id));
                        })
                        .await;
                    }
                    Err(e) => {
                        log::error!("Run spawn_racer system: {e}");
                    }
                }
            }
        }
    });
}

fn drive_thread(runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(DriveCommand(tx));

    runtime.spawn_background_task(|mut ctx| async move {
        // Get world contract
        // TODO: Can it be added as Resource or Component? If yes, we don't need the mpsc channel.
        // also how can I workaround &account ownership issue?
        let url = Url::parse(configs::JSON_RPC_ENDPOINT).unwrap();
        let account_address = FieldElement::from_str(configs::ACCOUNT_ADDRESS).unwrap();
        let account = SingleOwnerAccount::new(
            JsonRpcClient::new(HttpTransport::new(url)),
            LocalWallet::from_signing_key(SigningKey::from_secret_scalar(
                FieldElement::from_str(configs::ACCOUNT_SECRET_KEY).unwrap(),
            )),
            account_address,
            cairo_short_string_to_felt("KATANA").unwrap(),
        );

        let world_address = FieldElement::from_str(configs::WORLD_ADDRESS).unwrap();
        let block_id = BlockId::Tag(BlockTag::Latest);

        let world = WorldContract::new(world_address, &account);

        let drive_system = world.system("drive", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let dojo_ids = ctx
                .run_on_main_thread(move |ctx| {
                    let mut state: SystemState<Query<&Car>> = SystemState::new(ctx.world);
                    let query = state.get(ctx.world);

                    query
                        .iter()
                        .map(|car| car.dojo_id)
                        .collect::<Vec<FieldElement>>()
                })
                .await;

            for dojo_id in dojo_ids.iter() {
                if let Err(e) = drive_system.execute(vec![*dojo_id]).await {
                    log::error!("Run drive system: {e}");
                }
            }
        }
    });
}

fn update_vehicle_thread(runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(UpdateVehicleCommand(tx));

    runtime.spawn_background_task(|ctx| async move {
        let url = Url::parse(configs::JSON_RPC_ENDPOINT).unwrap();
        let account_address = FieldElement::from_str(configs::ACCOUNT_ADDRESS).unwrap();
        let account = SingleOwnerAccount::new(
            JsonRpcClient::new(HttpTransport::new(url)),
            LocalWallet::from_signing_key(SigningKey::from_secret_scalar(
                FieldElement::from_str(configs::ACCOUNT_SECRET_KEY).unwrap(),
            )),
            account_address,
            cairo_short_string_to_felt("KATANA").unwrap(),
        );

        let world_address = FieldElement::from_str(configs::WORLD_ADDRESS).unwrap();
        let block_id = BlockId::Tag(BlockTag::Latest);

        let world = WorldContract::new(world_address, &account);

        let vehicle_component = world.component("Vehicle", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let dojo_ids = get_dojo_ids(ctx.clone()).await;

            for id in dojo_ids.iter() {
                match vehicle_component
                    .entity(FieldElement::ZERO, vec![*id], block_id)
                    .await
                {
                    Ok(vehicle) => {
                        log::info!("Vehicle: {vehicle:#?}");
                        let dojo_x = to_f32(vehicle[0]);
                        let dojo_y = to_f32(vehicle[2]);

                        // log::info!(
                        //     "New Position: Vehicle({}), {{x: {}, y: {}}}",
                        //     id,
                        //     dojo_x,
                        //     dojo_y
                        // );

                        let (new_x, new_y) = to_road_position(dojo_x, dojo_y);

                        update_position::<Car>(ctx.clone(), new_x, new_y).await;
                    }
                    Err(e) => {
                        log::error!("Query `Vehicle` component: {e}");
                    }
                }
            }
        }
    });
}

fn update_enemies_thread(runtime: ResMut<TokioTasksRuntime>, mut commands: Commands) {
    let (tx, mut rx) = mpsc::channel::<()>(8);
    commands.insert_resource(UpdateEnemiesCommand(tx));

    runtime.spawn_background_task(|ctx| async move {
        let url = Url::parse(configs::JSON_RPC_ENDPOINT).unwrap();
        let account_address = FieldElement::from_str(configs::ACCOUNT_ADDRESS).unwrap();
        let account = SingleOwnerAccount::new(
            JsonRpcClient::new(HttpTransport::new(url)),
            LocalWallet::from_signing_key(SigningKey::from_secret_scalar(
                FieldElement::from_str(configs::ACCOUNT_SECRET_KEY).unwrap(),
            )),
            account_address,
            cairo_short_string_to_felt("KATANA").unwrap(),
        );

        let world_address = FieldElement::from_str(configs::WORLD_ADDRESS).unwrap();
        let block_id = BlockId::Tag(BlockTag::Latest);

        let world = WorldContract::new(world_address, &account);

        let position_component = world.component("Position", block_id).await.unwrap();

        while let Some(_) = rx.recv().await {
            let dojo_ids = get_dojo_ids(ctx.clone()).await;

            for id in dojo_ids {
                match position_component
                    .entity(FieldElement::ZERO, vec![id], block_id)
                    .await
                {
                    Ok(position) => {
                        // log::info!("{position:#?}");

                        let dojo_x = to_f32(position[0]);
                        let dojo_y = to_f32(position[1]);

                        // log::info!(
                        //     "Update Position: Enemy({}), {{x: {}, y: {}}}",
                        //     id,
                        //     dojo_x,
                        //     dojo_y
                        // );

                        let new_x = configs::WINDOW_WIDTH / 2.0
                            + (dojo_x * configs::WINDOW_WIDTH / configs::DOJO_GRID_WIDTH);
                        let new_y = configs::WINDOW_HEIGHT / 2.0
                            + (dojo_y * configs::WINDOW_HEIGHT / configs::DOJO_GRID_HEIGHT);

                        update_position::<Enemy>(ctx.clone(), new_x, new_y).await;
                    }
                    Err(e) => {
                        log::error!("Query `Position` component: {e}");
                    }
                }
            }
        }
    });
}

#[derive(Resource)]
pub struct SpawnRacersCommand(mpsc::Sender<()>);

// TODO: derive macro?
impl SpawnRacersCommand {
    pub fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
struct DriveCommand(mpsc::Sender<()>);

// TODO: derive macro?
impl DriveCommand {
    fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
struct UpdateVehicleCommand(mpsc::Sender<()>);

impl UpdateVehicleCommand {
    fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}

#[derive(Resource)]
pub struct UpdateEnemiesCommand(mpsc::Sender<()>);

impl UpdateEnemiesCommand {
    pub fn try_send(&self) -> Result<(), mpsc::error::TrySendError<()>> {
        self.0.try_send(())
    }
}
async fn get_dojo_ids(mut ctx: TaskContext) -> Vec<FieldElement> {
    ctx.run_on_main_thread(|ctx| {
        let mut state: SystemState<Query<&Car>> = SystemState::new(ctx.world);
        let query = state.get(ctx.world);

        query
            .iter()
            .map(|car| car.dojo_id)
            .collect::<Vec<FieldElement>>()
    })
    .await
}

fn to_f32(el: FieldElement) -> f32 {
    el.to_big_decimal(19).to_f32().unwrap()
}

async fn update_position<T>(mut ctx: TaskContext, x: f32, y: f32)
where
    T: Component,
{
    ctx.run_on_main_thread(move |ctx| {
        let mut state: SystemState<Query<&mut Transform, With<Enemy>>> =
            SystemState::new(ctx.world);
        let mut query = state.get_mut(ctx.world);
        for mut transform in query.iter_mut() {
            transform.translation.x = x;
            transform.translation.y = y;
        }
    })
    .await
}

// TODO
fn to_road_position(x: f32, y: f32) -> (f32, f32) {
    let rx = configs::WINDOW_WIDTH / 2.0 - 30.0;
    let ry = configs::ROAD_SPRITE_H / 2.0 * configs::SPRITE_SCALE_FACTOR
        + (configs::ROAD_SPRITE_H * configs::SPRITE_SCALE_FACTOR) * configs::NUM_ROAD_TILES as f32;

    let new_x = configs::WINDOW_WIDTH + (x * rx / configs::DOJO_GRID_WIDTH);
    let new_y = configs::WINDOW_HEIGHT + (y * ry / configs::DOJO_GRID_HEIGHT);

    (new_x, new_y)
}
