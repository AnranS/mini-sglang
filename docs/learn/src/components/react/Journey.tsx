import { useState, type KeyboardEvent, type ReactNode } from 'react';

export type JourneyRef = { label: string; loc: string; url: string; title: string };
export type JourneyStep = {
	lane: number;
	title: string;
	box: string;
	fn: string;
	detail: string;
	edge?: string;
	refs: JourneyRef[];
};
export type JourneyLane = { name: string; sub: string };
export type JourneyLoop = { from: number; to: number; label: string };

const SVG_W = 600;
const LANE_W = 150;
const BOX_W = 128;
const BOX_H = 36;
const BOX_DX = 11;
const TOP = 62;
const GAP = 16;
const LABEL_GAP = 30;

const laneX = (lane: number) => lane * LANE_W;
const midX = (lane: number) => lane * LANE_W + LANE_W / 2;

/** Renders `code` spans inside otherwise plain text. */
export function inline(text: string): ReactNode[] {
	return text.split(/(`[^`]+`)/).map((part, i) =>
		part.startsWith('`') && part.endsWith('`') ? <code key={i}>{part.slice(1, -1)}</code> : part,
	);
}

export default function Journey({
	lanes,
	steps,
	loop,
}: {
	lanes: JourneyLane[];
	steps: JourneyStep[];
	loop: JourneyLoop;
}) {
	const [idx, setIdx] = useState(0);
	const step = steps[idx];

	const stepY = steps.reduce<number[]>((acc, _s, i) => {
		acc.push(i === 0 ? TOP : acc[i - 1] + BOX_H + (steps[i - 1].edge ? LABEL_GAP : GAP));
		return acc;
	}, []);
	const svgH = stepY[stepY.length - 1] + BOX_H + 16;

	const loopLane = steps[loop.from].lane;
	const boxLeft = laneX(loopLane) + BOX_DX;
	const loopX = boxLeft - 9;
	const yA = stepY[loop.from] + BOX_H / 2;
	const yB = stepY[loop.to] + BOX_H / 2;

	const go = (i: number) => setIdx(Math.min(Math.max(i, 0), steps.length - 1));
	const onKey = (event: KeyboardEvent) => {
		if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
			event.preventDefault();
			go(idx + 1);
		} else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
			event.preventDefault();
			go(idx - 1);
		}
	};

	return (
		<div className="journey not-content">
			<svg
				className="journey-svg"
				viewBox={`0 0 ${SVG_W} ${svgH}`}
				role="group"
				aria-label="一个请求在四类执行单元之间的流转，方向键可以切换步骤"
				onKeyDown={onKey}
			>
				<defs>
					<marker
						id="journey-arrow"
						viewBox="0 0 8 8"
						refX={7}
						refY={4}
						markerWidth={7}
						markerHeight={7}
						markerUnits="userSpaceOnUse"
						orient="auto"
					>
						<path d="M0,0 L8,4 L0,8 z" className="journey-arrowhead" />
					</marker>
				</defs>

				{lanes.map((lane, k) => (
					<g key={lane.name}>
						{k % 2 === 0 ? (
							<rect className="journey-lane-bg" x={laneX(k)} y={0} width={LANE_W} height={svgH} rx={6} />
						) : null}
						<text className="journey-lane-name" x={midX(k)} y={22} textAnchor="middle">
							{lane.name}
						</text>
						<text className="journey-lane-sub" x={midX(k)} y={38} textAnchor="middle">
							{lane.sub}
						</text>
					</g>
				))}
				<line className="journey-divider" x1={0} x2={SVG_W} y1={50} y2={50} />

				{steps.slice(0, -1).map((a, i) => {
					const b = steps[i + 1];
					const x1 = midX(a.lane);
					const y1 = stepY[i] + BOX_H;
					const x2 = midX(b.lane);
					const y2 = stepY[i + 1] - 1;
					const my = y1 + 18;
					const d =
						a.lane === b.lane
							? `M${x1},${y1} L${x2},${y2}`
							: `M${x1},${y1} L${x1},${my} L${x2},${my} L${x2},${y2}`;
					return (
						<g key={`edge-${i}`}>
							<path className="journey-edge" d={d} markerEnd="url(#journey-arrow)" />
							{a.edge ? (
								<text className="journey-edge-label" x={(x1 + x2) / 2} y={y1 + 13} textAnchor="middle">
									{a.edge}
								</text>
							) : null}
						</g>
					);
				})}

				<path
					className="journey-edge journey-loop"
					d={`M${boxLeft},${yA} L${loopX},${yA} L${loopX},${yB} L${boxLeft - 1},${yB}`}
					markerEnd="url(#journey-arrow)"
				/>
				<text
					className="journey-edge-label"
					transform={`translate(${loopX - 5}, ${(yA + yB) / 2}) rotate(-90)`}
					textAnchor="middle"
				>
					{loop.label}
				</text>

				{steps.map((s, i) => {
					const x = laneX(s.lane) + BOX_DX;
					const y = stepY[i];
					return (
						<g
							key={s.title}
							className="journey-box"
							data-active={i === idx ? '' : undefined}
							role="button"
							tabIndex={0}
							aria-pressed={i === idx}
							aria-label={`第 ${i + 1} 步：${s.title}`}
							onClick={() => go(i)}
							onKeyDown={(event) => {
								if (event.key === 'Enter' || event.key === ' ') {
									event.preventDefault();
									go(i);
								}
							}}
						>
							<rect x={x} y={y} width={BOX_W} height={BOX_H} rx={6} />
							<text className="journey-box-title" x={x + 9} y={y + 15}>
								{`${i + 1} · ${s.title}`}
							</text>
							<text className="journey-box-fn" x={x + 9} y={y + 29}>
								{s.box}
							</text>
						</g>
					);
				})}
			</svg>

			<div className="journey-panel" aria-live="polite">
				<p className="journey-kicker">
					第 {idx + 1} / {steps.length} 步 · {lanes[step.lane].name}
				</p>
				<h3 className="journey-title">{step.title}</h3>
				<pre className="journey-fn">
					<code>{step.fn}</code>
				</pre>
				<p className="journey-detail">{inline(step.detail)}</p>
				<p className="journey-label">相关代码</p>
				<ul className="journey-refs">
					{step.refs.map((r) => (
						<li key={r.url + r.label}>
							<a className="src-ref" href={r.url} title={r.title} target="_blank" rel="noopener">
								<code>{r.label}</code>
								{r.loc ? <span className="src-ref-loc">{r.loc}</span> : null}
							</a>
						</li>
					))}
				</ul>
				<div className="journey-nav">
					<button type="button" disabled={idx === 0} onClick={() => go(idx - 1)}>
						上一步
					</button>
					<button type="button" className="primary" disabled={idx === steps.length - 1} onClick={() => go(idx + 1)}>
						下一步
					</button>
				</div>
			</div>
		</div>
	);
}
