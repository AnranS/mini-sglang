import { STATUS_LABEL, useProgress } from '../../lib/progress';

export type RoadmapStage = {
	id: string;
	n: number;
	title: string;
	time: string;
	summary: string;
	href: string;
};

export default function Roadmap({ stages }: { stages: RoadmapStage[] }) {
	const progress = useProgress();
	const done = stages.filter((s) => progress[s.id] === 'done').length;
	const next = stages.find((s) => progress[s.id] !== 'done');
	return (
		<div className="roadmap not-content">
			<div className="roadmap-progress">
				<div className="roadmap-bar" aria-hidden="true">
					<span style={{ width: `${(done / stages.length) * 100}%` }} />
				</div>
				<p>
					已完成 {done} / {stages.length} 个阶段
					{next ? (
						<>
							，下一站：
							<a href={next.href}>
								阶段 {next.n} · {next.title}
							</a>
						</>
					) : (
						'，全部完成，去挑一个毕业项目吧'
					)}
				</p>
			</div>
			<ol className="roadmap-list">
				{stages.map((s) => {
					const status = progress[s.id] ?? 'todo';
					return (
						<li key={s.id} className="roadmap-item" data-status={status}>
							<a href={s.href} className="roadmap-link">
								<span className="roadmap-num" aria-hidden="true">
									{s.n}
								</span>
								<span className="roadmap-body">
									<span className="roadmap-title">
										<span className="sr-only">阶段 {s.n}：</span>
										{s.title}
									</span>
									<span className="roadmap-text">{s.summary}</span>
								</span>
								<span className="roadmap-meta">
									<span>{s.time}</span>
									<span className="roadmap-state">{STATUS_LABEL[status]}</span>
								</span>
							</a>
						</li>
					);
				})}
			</ol>
		</div>
	);
}
