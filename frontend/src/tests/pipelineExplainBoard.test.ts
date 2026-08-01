import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount, tick, unmount } from 'svelte';

const lifecycleMocks = vi.hoisted(() => ({
	explainPipeline: vi.fn(),
}));

const realtimeMocks = vi.hoisted(() => ({
	createRealtimeRefresh: vi.fn(() => ({ start: vi.fn(), stop: vi.fn() })),
}));

vi.mock('$lib/api/lifecycle', () => lifecycleMocks);
vi.mock('$lib/utils/realtime', () => realtimeMocks);

import PipelineExplainBoard from '../lib/components/lifecycle/PipelineExplainBoard.svelte';

type MountedComponent = ReturnType<typeof mount>;

async function flush(): Promise<void> {
	await Promise.resolve();
	await tick();
	await Promise.resolve();
	await tick();
}

function fleetPayload() {
	return {
		ok: true,
		generated_at: '2026-07-29T00:00:00+00:00',
		pipeline_preset: 'default',
		stages: ['quick_screen', 'gauntlet', 'paper', 'live_graduated'],
		counts: {
			by_stage: { gauntlet: 1, paper: 1 },
			by_status: { waiting_evidence: 1, ready: 1 },
		},
		truncated: false,
		errors: [],
		strategies: [
			{
				id: 'S1',
				display_id: 'S00001',
				name: 'BTC breakout',
				symbol: 'BTC/USDT',
				timeframe: '1h',
				type: 'breakout',
				stage: 'gauntlet',
				stage_label: 'Gauntlet',
				stage_changed_at: '2026-07-17T00:00:00+00:00',
				days_in_stage: 12.3,
				demotion_count: 0,
				status: 'waiting_evidence',
				promotable: false,
				gate_reason: 'Missing passing persisted artifact rows for: walk_forward',
				blockers: [
					{
						reason: 'Missing passing persisted artifact rows for: walk_forward',
						code: 'missing_evidence',
						kind: 'evidence',
						source: 'gauntlet_gate',
						action: { key: 'run_validation_suite', label: 'Run the gauntlet validation suite' },
					},
				],
				next_action: { key: 'run_validation_suite', label: 'Run the gauntlet validation suite' },
				next_transition: {
					to_stage: 'paper',
					label: 'Gauntlet → Paper trading',
					trigger: 'Automatic — promotes once every required validation test passes',
				},
				evidence: {
					last_backtest_at: '2026-07-20T00:00:00+00:00',
					last_backtest_age_days: 9.0,
					validation_tests: {
						monte_carlo: {
							status: 'passed',
							verdict: 'PASS',
							at: '2026-07-21T00:00:00+00:00',
							age_days: 8.0,
							stale: true,
							stale_engine: null,
						},
					},
				},
				readiness_steps: [],
				gauntlet: {
					workflow_status: 'pending',
					current_step: null,
					required_tests: ['walk_forward', 'param_jitter'],
					missing_required: ['walk_forward'],
					tests_passed: 1,
					tests_total: 5,
					composite_robustness_score: 42.0,
					min_robustness_score: 30.0,
				},
				pending_approval: null,
				last_rejection: {
					gate: 'gauntlet',
					reason_code: 'missing_evidence',
					reason_text: 'Missing passing persisted artifact rows for: walk_forward',
					at: '2026-07-28T00:00:00+00:00',
					age_days: 1.0,
				},
				rejections_in_stage: 3,
			},
			{
				id: 'S2',
				display_id: 'S00002',
				name: 'ETH momo',
				symbol: 'ETH/USDT',
				timeframe: '4h',
				type: 'momentum',
				stage: 'paper',
				stage_label: 'Paper trading',
				stage_changed_at: '2026-07-01T00:00:00+00:00',
				days_in_stage: 28.0,
				demotion_count: 0,
				status: 'ready',
				promotable: true,
				gate_reason: 'Passed paper forward-proof gate (live graduated cap: 25% allocation)',
				blockers: [],
				next_action: { key: 'promote', label: 'Ready — promote to Live (graduated)' },
				next_transition: {
					to_stage: 'live_graduated',
					label: 'Paper trading → Live (graduated)',
					trigger: 'Gated — the strict paper→live gate must pass',
				},
				evidence: {
					paper: {
						paper_duration: { current: 28, threshold: 14, unit: 'days' },
						paper_trades: { current: 31, threshold: 10, unit: 'trades' },
					},
				},
				readiness_steps: [],
				gauntlet: null,
				pending_approval: null,
				last_rejection: null,
				rejections_in_stage: 0,
			},
		],
	};
}

describe('PipelineExplainBoard', () => {
	let target: HTMLDivElement;
	let app: MountedComponent | null = null;

	beforeEach(() => {
		target = document.createElement('div');
		document.body.appendChild(target);
		lifecycleMocks.explainPipeline.mockReset();
	});

	afterEach(() => {
		if (app) {
			unmount(app);
			app = null;
		}
		target.remove();
		vi.clearAllMocks();
	});

	it('renders stage columns with cards, status badges, blockers, and actions', async () => {
		lifecycleMocks.explainPipeline.mockResolvedValue(fleetPayload());

		app = mount(PipelineExplainBoard, { target });
		await flush();

		expect(target.querySelector('[data-testid="pipeline-explain-board"]')).not.toBeNull();
		for (const stage of ['quick_screen', 'gauntlet', 'paper', 'live_graduated']) {
			expect(target.querySelector(`[data-testid="explain-column-${stage}"]`)).not.toBeNull();
		}

		expect(target.querySelector('[data-testid="explain-card-S1"]')).not.toBeNull();
		expect(target.querySelector('[data-testid="explain-status-S1"]')?.textContent).toContain('WAITING ON EVIDENCE');
		expect(target.querySelector('[data-testid="explain-blocker-S1"]')?.textContent).toContain(
			'Missing passing persisted artifact rows for: walk_forward'
		);
		expect(target.querySelector('[data-testid="explain-action-S1"]')?.textContent).toContain(
			'Run the gauntlet validation suite'
		);
		expect(target.querySelector('[data-testid="explain-card-S1"]')?.textContent).toContain('in stage 12d');
		expect(target.querySelector('[data-testid="explain-card-S1"]')?.textContent).toContain('3 rejections');

		expect(target.querySelector('[data-testid="explain-status-S2"]')?.textContent).toContain('READY');
		expect(target.querySelector('[data-testid="explain-action-S2"]')?.textContent).toContain('promote to Live');
	});

	it('expands a card to show next transition, evidence ages, and staleness', async () => {
		lifecycleMocks.explainPipeline.mockResolvedValue(fleetPayload());

		app = mount(PipelineExplainBoard, { target });
		await flush();

		const toggle = target.querySelector('[data-testid="explain-toggle-S1"]') as HTMLButtonElement;
		expect(toggle).not.toBeNull();
		toggle.click();
		await flush();

		const detail = target.querySelector('[data-testid="explain-detail-S1"]');
		expect(detail).not.toBeNull();
		expect(detail?.textContent).toContain('Gauntlet → Paper trading');
		expect(detail?.textContent).toContain('monte_carlo');
		expect(detail?.textContent).toContain('STALE');
		expect(detail?.textContent).toContain('Last backtest 9d ago');
		expect(detail?.textContent).toContain('Last rejection at the gauntlet gate 1d ago');
	});

	it('surfaces a load error', async () => {
		lifecycleMocks.explainPipeline.mockRejectedValue(new Error('backend down'));

		app = mount(PipelineExplainBoard, { target });
		await flush();

		expect(target.querySelector('[data-testid="explain-error"]')?.textContent).toContain('backend down');
	});
});
